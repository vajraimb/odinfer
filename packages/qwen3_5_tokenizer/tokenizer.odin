/* BPE tokenizer for Qwen3.5 (Ornith-1.0-9B). Same GPT-2 byte-level BPE scheme as
   the Qwen3 tokenizer; differs only in vocab/merges data, special token set,
   and uses a hash map for O(1) merge lookup (247k merges rules out linear scan). */

package qwen3_5_tokenizer

import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"
import "core:testing"

MAX_TOKENS :: 1024
MAX_TOKEN_LEN :: 32
// Ceiling for a Qwen3.5 vocab file. The embedding table is padded to
// 248320; the embedded vocab.txt is 248044 BPE tokens + 33 specials
// (ids 248044..248076). A file of 248320 lines must load, not stop at
// 248077 and look like the embedded vocab.
MAX_VOCAB :: 248320

MERGE_NOT_FOUND :: 1_000_000_000
MAX_MERGES :: 247587

EMBEDDED_VOCAB :: string(#load("vocab.txt"))
EMBEDDED_MERGES :: string(#load("merges.txt"))

Tokenizer :: struct {
	vocab:          [dynamic]string,
	vocab_lookup:   map[string]int,
	merge_lookup:   map[string]int, // key = "<left>\n<right>" -> rank
	unicode_to_byte: map[string]u8, // per-instance (avoids races across parallel users)
}

byte_to_unicode: [256]u32
unicode_bytes: [256][5]u8

SPECIAL_TOKENS :: []string {
	"<|im_start|>",
	"<|im_end|>",
	"<|endoftext|>",
	"<think>",
	"</think>",
	"<tool_call>",
	"</tool_call>",
	"<tool_response>",
	"</tool_response>",
	"<|object_ref_start|>",
	"<|object_ref_end|>",
	"<|box_start|>",
	"<|box_end|>",
	"<|quad_start|>",
	"<|quad_end|>",
	"<|vision_start|>",
	"<|vision_end|>",
	"<|vision_pad|>",
	"<|image_pad|>",
	"<|video_pad|>",
	"<|fim_prefix|>",
	"<|fim_middle|>",
	"<|fim_suffix|>",
	"<|fim_pad|>",
	"<|repo_name|>",
	"<|file_sep|>",
	"<|audio_start|>",
	"<|audio_end|>",
	"<|audio_pad|>",
	"<tts_pad>",
	"<tts_text_bos>",
	"<tts_text_eod>",
	"<tts_text_bos_single>",
}

init_byte_unicode_map :: proc(t: ^Tokenizer) {
	t.unicode_to_byte = make(map[string]u8)
	n := 0
	for b in 0 ..< 256 {
		if (b >= 33 && b <= 126) ||
		   (b >= 161 && b <= 172) ||
		   (b >= 174 && b <= 255) {
			byte_to_unicode[b] = u32(b)
		} else {
			byte_to_unicode[b] = u32(256 + n)
			n += 1
		}

		cp := byte_to_unicode[b]
		if cp < 128 {
			unicode_bytes[b][0] = u8(cp)
			unicode_bytes[b][1] = 0
		} else if cp < 2048 {
			unicode_bytes[b][0] = 0xC0 | u8(cp >> 6)
			unicode_bytes[b][1] = 0x80 | u8(cp & 0x3F)
			unicode_bytes[b][2] = 0
		} else {
			unicode_bytes[b][0] = 0xE0 | u8(cp >> 12)
			unicode_bytes[b][1] = 0x80 | u8((cp >> 6) & 0x3F)
			unicode_bytes[b][2] = 0x80 | u8(cp & 0x3F)
			unicode_bytes[b][3] = 0
		}

		end := 0
		for end < len(unicode_bytes[b]) && unicode_bytes[b][end] != 0 {
			end += 1
		}
		key := strings.clone(string(unicode_bytes[b][:end]))
		t.unicode_to_byte[key] = u8(b)
	}
}

load_vocab_data :: proc(t: ^Tokenizer, data: string) {
	t.vocab = make([dynamic]string, 0, MAX_VOCAB)
	t.vocab_lookup = make(map[string]int)

	lines := strings.split_lines(data)
	defer delete(lines)
	n := len(lines)
	// A trailing newline is not a token. Counting it used to trip the old
	// 248077 ceiling and drop the load back onto the embedded vocab.
	if n > 0 && lines[n - 1] == "" {
		n -= 1
	}
	for i in 0 ..< n {
		line := lines[i]
		if len(t.vocab) >= MAX_VOCAB {
			fmt.eprintf("vocab exceeds MAX_VOCAB (%d); further tokens were not loaded\n", MAX_VOCAB)
			break
		}
		id := len(t.vocab)
		token := strings.clone(line)
		t.vocab_lookup[token] = id
		append(&t.vocab, token)
	}
}

load_merges_data :: proc(t: ^Tokenizer, data: string) {
	t.merge_lookup = make(map[string]int, MAX_MERGES)
	buf: [512]u8
	rank := 0

	for line in strings.split_lines(data) {
		if len(line) == 0 || line[0] == '#' do continue
		parts := strings.fields(line)
		if len(parts) < 2 do continue
		// owned composite key so the map retains it; lookup builds the same key
		// in a stack buffer (no allocation) via get_merge_rank.
		n := copy(buf[:], transmute([]u8)parts[0])
		buf[n] = '\n'
		n += 1
		n += copy(buf[n:], transmute([]u8)parts[1])
		key := strings.clone(string(buf[:n]))
		t.merge_lookup[key] = rank
		rank += 1
	}
}

load_vocab_file :: proc(t: ^Tokenizer, path: string) -> bool {
	data, err := os.read_entire_file_from_path(path, context.temp_allocator)
	if err != os.ERROR_NONE do return false
	load_vocab_data(t, string(data))
	return true
}

load_merges_file :: proc(t: ^Tokenizer, path: string) -> bool {
	data, err := os.read_entire_file_from_path(path, context.temp_allocator)
	if err != os.ERROR_NONE do return false
	load_merges_data(t, string(data))
	return true
}

// Prefer vocab.txt / merges.txt in the current directory; fall back to data
// embedded at compile time (#load) so a lone binary still works for distribution.
build_tokenizer :: proc(t: ^Tokenizer, vocab_path: string = "vocab.txt", merges_path: string = "merges.txt") {
	init_byte_unicode_map(t)

	vocab_ok := load_vocab_file(t, vocab_path)
	merges_ok := load_merges_file(t, merges_path)

	if !vocab_ok {
		load_vocab_data(t, EMBEDDED_VOCAB)
	}
	if !merges_ok {
		load_merges_data(t, EMBEDDED_MERGES)
	}
	if !vocab_ok || !merges_ok {
		fmt.eprintln("tokenizer: using embedded vocab/merges (place vocab.txt merges.txt in cwd to override)")
	}
}

verify_tokenizer :: proc(t: ^Tokenizer) -> bool {
	test := "<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
	ids, err := encode(t, test)
	if err != nil || len(ids) == 0 do return false
	for id in ids {
		if id == 0 do return false
	}
	return true
}

free_tokenizer :: proc(t: ^Tokenizer) {
	for s in t.vocab {
		delete(s)
	}
	delete(t.vocab)
	delete(t.vocab_lookup)
	for k, _ in t.merge_lookup {
		delete(k)
	}
	delete(t.merge_lookup)
	delete(t.unicode_to_byte)
}

// O(1) merge-rank lookup. Builds the composite key in a stack buffer (no
// allocation); Odin hashes string map keys by content so a non-owned string is
// fine for lookup.
get_merge_rank :: proc(t: ^Tokenizer, left, right: string) -> int {
	buf: [512]u8
	n := copy(buf[:], transmute([]u8)left)
	buf[n] = '\n'
	n += 1
	n += copy(buf[n:], transmute([]u8)right)
	if r, ok := t.merge_lookup[string(buf[:n])]; ok {
		return r
	}
	return MERGE_NOT_FOUND
}

match_special_token :: proc(t: ^Tokenizer, str: string) -> (id: int, match_len: int, ok: bool) {
	for special in SPECIAL_TOKENS {
		if strings.has_prefix(str, special) {
			if token_id, found := t.vocab_lookup[special]; found {
				return token_id, len(special), true
			}
		}
	}
	return -1, 0, false
}

is_special_pair :: proc(a, b: string) -> bool {
	return len(a) > 0 && a[0] == '<' && strings.contains_rune(a, '|') &&
	       len(b) > 0 && b[0] == '<' && strings.contains_rune(b, '|')
}

// One symbol in the BPE chain. `start` is the byte offset of its first
// byte, which is the leftmost-wins tie break of the old scan.
Bpe_Node :: struct {
	piece: string,
	prev:  int,
	next:  int,
	ver:   int,
	start: int,
}

Bpe_Cand :: struct {
	rank:  int,
	start: int,
	left:  int,
	right: int,
	lv:    int,
	rv:    int,
}

@(private)
bpe_before :: proc(a, b: Bpe_Cand) -> bool {
	if a.rank != b.rank {
		return a.rank < b.rank
	}
	return a.start < b.start
}

@(private)
bpe_push :: proc(h: ^[dynamic]Bpe_Cand, c: Bpe_Cand) {
	append(h, c)
	i := len(h) - 1
	for i > 0 {
		p := (i - 1) / 2
		if !bpe_before(h[i], h[p]) {
			break
		}
		h[i], h[p] = h[p], h[i]
		i = p
	}
}

@(private)
bpe_pop :: proc(h: ^[dynamic]Bpe_Cand) -> (Bpe_Cand, bool) {
	if len(h) == 0 {
		return {}, false
	}
	top := h[0]
	last := len(h) - 1
	h[0] = h[last]
	resize(h, last)
	i := 0
	for {
		l := i * 2 + 1
		if l >= len(h) {
			break
		}
		s := l
		r := l + 1
		if r < len(h) && bpe_before(h[r], h[l]) {
			s = r
		}
		if !bpe_before(h[s], h[i]) {
			break
		}
		h[i], h[s] = h[s], h[i]
		i = s
	}
	return top, true
}

// Same merges as the old full scan (lowest rank, leftmost on a tie), in
// O(n log n) instead of one pass over the whole chain per merge. A 5k-token
// string was 3.7 s; 15k tokens was about a minute.
encode :: proc(t: ^Tokenizer, text: string, allocator := context.allocator) -> ([]int, mem.Allocator_Error) {
	nodes := make([dynamic]Bpe_Node, 0, len(text) + 8)
	owned := make([dynamic][]u8, 0, 64)
	defer {
		for b in owned {
			delete(b)
		}
		delete(owned)
		delete(nodes)
	}

	prev := -1
	i := 0
	for i < len(text) {
		piece: string
		start := i
		if id, match_len, found := match_special_token(t, text[i:]); found {
			piece = t.vocab[id]
			i += match_len
		} else {
			b := text[i]
			end := 0
			for end < len(unicode_bytes[b]) && unicode_bytes[b][end] != 0 {
				end += 1
			}
			piece = string(unicode_bytes[b][:end])
			i += 1
		}
		idx := len(nodes)
		append(&nodes, Bpe_Node{piece = piece, prev = prev, next = -1, ver = 0, start = start})
		if prev >= 0 {
			nodes[prev].next = idx
		}
		prev = idx
	}

	heap := make([dynamic]Bpe_Cand, 0, len(nodes))
	defer delete(heap)
	push_pair :: proc(h: ^[dynamic]Bpe_Cand, nodes: ^[dynamic]Bpe_Node, tok: ^Tokenizer, left: int) {
		if left < 0 {
			return
		}
		right := nodes[left].next
		if right < 0 {
			return
		}
		rank := get_merge_rank(tok, nodes[left].piece, nodes[right].piece)
		if rank == MERGE_NOT_FOUND {
			return
		}
		bpe_push(h, Bpe_Cand{
			rank = rank,
			start = nodes[left].start,
			left = left,
			right = right,
			lv = nodes[left].ver,
			rv = nodes[right].ver,
		})
	}
	for n in 0 ..< len(nodes) - 1 {
		push_pair(&heap, &nodes, t, n)
	}

	for {
		cand, ok := bpe_pop(&heap)
		if !ok {
			break
		}
		if nodes[cand.left].ver != cand.lv || nodes[cand.right].ver != cand.rv {
			continue
		}
		if nodes[cand.left].next != cand.right {
			continue
		}
		if is_special_pair(nodes[cand.left].piece, nodes[cand.right].piece) {
			break
		}
		left := nodes[cand.left]
		right := nodes[cand.right]
		buf := make([]u8, len(left.piece) + len(right.piece))
		copy(buf, left.piece)
		copy(buf[len(left.piece):], right.piece)
		append(&owned, buf)
		idx := len(nodes)
		append(&nodes, Bpe_Node{
			piece = string(buf),
			prev = left.prev,
			next = right.next,
			ver = 0,
			start = left.start,
		})
		nodes[cand.left].ver += 1
		nodes[cand.right].ver += 1
		if left.prev >= 0 {
			nodes[left.prev].next = idx
		}
		if right.next >= 0 {
			nodes[right.next].prev = idx
		}
		push_pair(&heap, &nodes, t, left.prev)
		push_pair(&heap, &nodes, t, idx)
	}

	count := 0
	head := -1
	for n in 0 ..< len(nodes) {
		if nodes[n].prev < 0 && nodes[n].ver == 0 {
			head = n
			break
		}
	}
	for at := head; at >= 0 && count < len(nodes); at = nodes[at].next {
		count += 1
	}
	result, err := make([]int, count, allocator)
	if err != nil {
		return nil, err
	}
	out := 0
	for at := head; at >= 0; at = nodes[at].next {
		if id, found := t.vocab_lookup[nodes[at].piece]; found {
			result[out] = id
		} else {
			result[out] = 0
		}
		out += 1
	}
	return result, nil
}

decode_token_id :: proc(t: ^Tokenizer, token_id: int, allocator := context.allocator) -> string {
	if token_id < 0 || token_id >= len(t.vocab) {
		return ""
	}
	encoded := t.vocab[token_id]

	builder: strings.Builder
	strings.builder_init(&builder, allocator)
	defer strings.builder_destroy(&builder)

	i := 0
	for i < len(encoded) {
		matched := false
		for length in 1 ..= 3 {
			if i + length > len(encoded) do break
			utf8_str := encoded[i:i + length]
			if b, ok := t.unicode_to_byte[utf8_str]; ok {
				strings.write_byte(&builder, b)
				i += length
				matched = true
				break
			}
		}
		if !matched do break
	}

	return strings.clone(strings.to_string(builder), allocator)
}

Token_Buffer :: struct {
	data: [dynamic]int,
}

build_token_buffer :: proc(tb: ^Token_Buffer, initial_capacity: int = 1024) {
	tb.data = make([dynamic]int, 0, initial_capacity)
}

free_token_buffer :: proc(tb: ^Token_Buffer) {
	delete(tb.data)
}

append_tokens :: proc(tb: ^Token_Buffer, tokens: []int) {
	append(&tb.data, ..tokens)
}

@(test)
encode_decode_roundtrip :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	text := "Hello, world!"
	ids, err := encode(&tok, text)
	testing.expect(t, err == nil, "encode failed")
	testing.expect(t, len(ids) > 0, "expected tokens")

	decoded_parts: [dynamic]string
	defer delete(decoded_parts)
	for id in ids {
		append(&decoded_parts, decode_token_id(&tok, id))
	}
	reconstructed := strings.concatenate(decoded_parts[:], context.temp_allocator)
	testing.expect(t, reconstructed == text, "roundtrip mismatch")
}

@(test)
encode_special_token_ids :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	prompt := "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
	ids, err := encode(&tok, prompt)
	testing.expect(t, err == nil, "encode failed")
	testing.expect(t, len(ids) > 0, "expected tokens")

	im_start, _ := tok.vocab_lookup["<|im_start|>"]
	im_end, _ := tok.vocab_lookup["<|im_end|>"]
	testing.expect(t, im_start == 248045, "im_start id")
	testing.expect(t, im_end == 248046, "im_end id")
	testing.expect(t, ids[0] == im_start, "first token should be im_start")
}

@(test)
encode_chinese :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	text := "你能做什么"
	ids, err := encode(&tok, text)
	testing.expect(t, err == nil, "encode failed")
	testing.expect(t, len(ids) > 0, "expected tokens")
	for id in ids {
		testing.expect(t, id != 0, "unknown token in chinese encode")
	}
}

@(test)
encode_chat_template :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	prompt := "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
	ids, err := encode(&tok, prompt)
	testing.expect(t, err == nil, "encode failed")
	for id in ids {
		if id == 0 {
			testing.expect(t, false, "token id 0 used for unknown token")
		}
	}
}

// Locks the heap merge to the old left-to-right scan. The repeated sentence
// is the cuda_smoke prompt body. 331 tokens used to take 15 ms and 5131
// tokens 3.7 s; both stay on this hash.
@(test)
encode_merge_order :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)
	body := "The quick brown fox jumps over the lazy dog. "
	reps := [?]int{33, 513}
	want := [?]u64{2918863925090883675, 13672901658331668571}
	for i in 0 ..< len(reps) {
		parts := make([]string, reps[i])
		for p in 0 ..< reps[i] {
			parts[p] = body
		}
		text := strings.concatenate(parts)
		delete(parts)
		ids, err := encode(&tok, text)
		delete(text)
		testing.expect(t, err == nil, "encode failed")
		h: u64 = 14695981039346656037
		for id in ids {
			h ~= u64(id)
			h *= 1099511628211
		}
		delete(ids)
		testing.expect(t, h == want[i], "merge order changed")
	}
}

@(test)
think_token_id :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	think_open, _ := tok.vocab_lookup["<think>"]
	think_close, _ := tok.vocab_lookup["</think>"]
	eot, _ := tok.vocab_lookup["<|endoftext|>"]
	testing.expect(t, think_open == 248068, "<think> id")
	testing.expect(t, think_close == 248069, "</think> id")
	testing.expect(t, eot == 248044, "<|endoftext|> id")
}
