/* End-to-end CUDA smoke: load a Qwen3.5 GGUF, prefill a short prompt, decode
   N tokens at temperature 0, print prefill and decode tok/s.

   odin run tools/cuda_smoke.odin -file -define:ODINFER_BACKEND="cuda" \
     -collection:ggml=packages -collection:infer=packages \
     -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
     -collection:sampler=packages -collection:tokenizer=packages \
     -collection:cuda=packages -- <model.gguf> [n_tokens] [max_ctx]

   Default model search is done by scripts/cuda_check.sh. This tool requires
   the path. n_tokens defaults to 16, max_ctx to 2048.
   ODINFER_CUDA_SMOKE_PREFILL=N builds a deterministic prompt and prefills
   N tokens (the short 2+2 prompt is the default). max_ctx must cover
   N plus the decoded tokens.
   ODINFER_CUDA_LOGITS=1 prints the top-5 logits at the first decode
   position and, after that token is fed back, at the second. EOS is
   248046. Compare a default run with ODINFER_CUDA_PREFILL_GEMM=0.
*/

package main

import q35 "qwen3_5:qwen3_5"
import sampler "sampler:sampler"
import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

EOS_QWEN3_5 :: 248046

logits_dump :: proc(logits: []f32, tag: string) {
	if os.get_env("ODINFER_CUDA_LOGITS", context.temp_allocator) != "1" || len(logits) == 0 {
		return
	}
	top_i: [5]int
	top_v: [5]f32
	for i in 0 ..< 5 {
		top_i[i] = -1
		top_v[i] = -1e30
	}
	for i in 0 ..< len(logits) {
		v := logits[i]
		if v <= top_v[4] {
			continue
		}
		slot := 4
		for slot > 0 && v > top_v[slot - 1] {
			top_v[slot] = top_v[slot - 1]
			top_i[slot] = top_i[slot - 1]
			slot -= 1
		}
		top_v[slot] = v
		top_i[slot] = i
	}
	eos_v: f32 = -1e30
	if EOS_QWEN3_5 < len(logits) {
		eos_v = logits[EOS_QWEN3_5]
	}
	fmt.eprintf("cuda_smoke: logits %s top5", tag)
	for i in 0 ..< 5 {
		fmt.eprintf(" %d:%.5f", top_i[i], top_v[i])
	}
	fmt.eprintfln(" eos %d:%.5f margin %.5f", EOS_QWEN3_5, eos_v, top_v[0] - eos_v)
}

main :: proc() {
	when !q35.GPU_CUDA {
		fmt.eprintln("cuda_smoke: rebuild with -define:ODINFER_BACKEND=\"cuda\"")
		os.exit(1)
	} else {
		if len(os.args) < 2 {
			fmt.eprintln("usage: cuda_smoke <model.gguf> [n_tokens] [max_ctx]")
			os.exit(1)
		}
		path := os.args[1]
		n_tokens := 16
		max_ctx := 2048
		if len(os.args) > 2 {
			if v, ok := strconv.parse_int(os.args[2]); ok && v > 0 {
				n_tokens = int(v)
			}
		}
		if len(os.args) > 3 {
			if v, ok := strconv.parse_int(os.args[3]); ok && v > 0 {
				max_ctx = int(v)
			}
		}
		engine, load_ok := q35.engine_load(path, q35.Engine_Opts{max_ctx = max_ctx, use_metal = true, num_threads = 4})
		if !load_ok || !engine.metal_ready {
			fmt.eprintln("cuda_smoke: engine_load failed (GPU init is required; METAL_FALLBACK is not a pass)")
			os.exit(1)
		}
		defer q35.engine_destroy(&engine)
		cfg := q35.engine_config(&engine)
		fmt.eprintfln("cuda_smoke: backend=%s dim=%d layers=%d vocab=%d ctx=%d",
			q35.gpu_backend_name(engine.metal_ready), cfg.dim, cfg.n_layers, cfg.vocab_size, cfg.seq_len)

		tok: tok35.Tokenizer
		tok35.build_tokenizer(&tok)
		defer tok35.free_tokenizer(&tok)
		prompt := "<|im_start|>user\nWhat is 2+2? Answer with one number.<|im_end|>\n<|im_start|>assistant\n"
		want_prefill := 0
		if s := os.get_env("ODINFER_CUDA_SMOKE_PREFILL", context.temp_allocator); s != "" {
			if v, ok := strconv.parse_int(s); ok && v > 1 {
				want_prefill = int(v)
			}
		}
		encoded: []int
		if want_prefill > 0 {
			// Repeat a fixed sentence so the token stream is deterministic
			// and old-vs-new text still compares. Slice to exactly N prefill
			// tokens plus the token that starts decode.
			need := want_prefill + 1
			if need > max_ctx {
				fmt.eprintfln("cuda_smoke: prefill %d plus the first decode token needs max_ctx >= %d", want_prefill, need)
				os.exit(1)
			}
			head := "<|im_start|>user\n"
			tail := "\nWhat is 2+2? Answer with one number.<|im_end|>\n<|im_start|>assistant\n"
			body := "The quick brown fox jumps over the lazy dog. "
			// One encode. The sentence is about 8 tokens and 45 bytes; size
			// the repeat from that and grow once if the estimate is short.
			reps := need
			built := false
			for attempt in 0 ..< 3 {
				parts := make([]string, reps + 2)
				parts[0] = head
				for i in 0 ..< reps {
					parts[1 + i] = body
				}
				parts[reps + 1] = tail
				prompt = strings.concatenate(parts)
				delete(parts)
				enc, eerr := tok35.encode(&tok, prompt)
				delete(prompt)
				if eerr != nil {
					fmt.eprintln("cuda_smoke: encode failed")
					os.exit(1)
				}
				if len(enc) >= need {
					encoded = enc
					built = true
					break
				}
				delete(enc)
				reps *= 2
			}
			if !built {
				fmt.eprintfln("cuda_smoke: could not build a %d-token prompt", want_prefill)
				os.exit(1)
			}
			encoded = encoded[:need]
		} else {
			enc, eerr := tok35.encode(&tok, prompt)
			if eerr != nil || len(enc) < 2 {
				fmt.eprintln("cuda_smoke: encode failed")
				os.exit(1)
			}
			encoded = enc
		}
		defer delete(encoded)

		fmt.eprintfln("cuda_smoke: bench prefill={} decode_req={} ctx={} temp=0", len(encoded) - 1, n_tokens, max_ctx)
		t0 := time.now()
		logits := q35.engine_forward_batch(&engine, encoded[:len(encoded) - 1], 0)
		prefill_s := time.duration_seconds(time.diff(t0, time.now()))
		prefill_n := len(encoded) - 1
		fmt.eprintfln("cuda_smoke: prefill %d tok in %.3f s (%.2f tok/s)", prefill_n, prefill_s, f64(prefill_n) / prefill_s)

		samp: sampler.Sampler
		sampler.build_sampler(&samp, cfg.vocab_size, 0, 1, 1)
		defer sampler.free_sampler(&samp)

		text: strings.Builder
		strings.builder_init(&text)
		defer strings.builder_destroy(&text)

		pos := prefill_n
		logits_dump(logits, "pos0")
		next := sampler.sample(&samp, logits)
		t1 := time.now()
		produced := 0
		for step in 0 ..< n_tokens {
			piece := tok35.decode_token_id(&tok, next)
			strings.write_string(&text, piece)
			delete(piece)
			produced += 1
			if next == EOS_QWEN3_5 {
				break
			}
			if pos >= cfg.seq_len - 1 {
				break
			}
			logits = q35.engine_forward(&engine, next, pos)
			pos += 1
			if produced == 1 {
				logits_dump(logits, "pos1")
			}
			next = sampler.sample(&samp, logits)
			_ = step
		}
		decode_s := time.duration_seconds(time.diff(t1, time.now()))
		rate := 0.0
		if decode_s > 0 {
			rate = f64(produced) / decode_s
		}
		fmt.eprintfln("cuda_smoke: decode %d tok in %.3f s (%.2f tok/s)", produced, decode_s, rate)
		fmt.eprintfln("cuda_smoke: text: %s", strings.to_string(text))
	}
}
