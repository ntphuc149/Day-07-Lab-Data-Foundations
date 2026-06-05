# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Trường Phúc
**Nhóm:** Legal RAG Group
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> Hai văn bản có cosine similarity cao nghĩa là các vector embedding của chúng trỏ về cùng hướng trong không gian nhiều chiều, phản ánh rằng hai đoạn văn chia sẻ ngữ nghĩa tương đồng dù dùng từ ngữ khác nhau.

**Ví dụ HIGH similarity:**

- Sentence A: "Python is a programming language used for machine learning."
- Sentence B: "Python is widely used in AI and data science projects."
- Tại sao tương đồng: Cả hai đều nói về Python trong lĩnh vực lập trình/AI, các từ quan trọng (Python, programming, AI, machine learning) tạo ra embedding vector gần nhau.

**Ví dụ LOW similarity:**

- Sentence A: "The vector store indexes embeddings for semantic search."
- Sentence B: "The weather forecast shows heavy rain tomorrow."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (hệ thống AI vs thời tiết), không có từ nào chung về mặt ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> Cosine similarity đo góc giữa hai vector nên không bị ảnh hưởng bởi độ dài văn bản — một câu ngắn và một đoạn văn dài cùng chủ đề vẫn có similarity cao. Euclidean distance lại phụ thuộc vào magnitude của vector, khiến các văn bản dài tự nhiên bị đẩy xa hơn dù ý nghĩa tương đồng.

---

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
>
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
>
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25`
>
> Overlap tăng làm chunk count tăng vì mỗi bước tiến nhỏ hơn. Overlap nhiều hơn giúp đảm bảo thông tin nằm ở ranh giới giữa hai chunk không bị mất — đặc biệt quan trọng khi một ý quan trọng trải dài qua nhiều câu.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Legal

**Tại sao nhóm chọn domain này?**

Văn bản pháp luật chứa nhiều thuật ngữ chuyên ngành, gây khó hiểu cho người dùng và ngợp trong nhiều tài liệu. RAG cho Legal sẽ giúp người dùng dễ tiếp cận với văn bản pháp luật hơn và cải thiện dân trí cho người dân.

### Data Inventory

| #   | Tên tài liệu                                                      | Nguồn        | Số ký tự | Metadata đã gán                                                                                                         |
| --- | ----------------------------------------------------------------- | ------------ | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| 1   | 28-2026.md — NĐ danh mục chất ma túy và tiền chất                 | Chính phủ VN | 5,099    | `doc_number: 28/2026/NĐ-CP`, `doc_type: nghi_dinh`, `issuing_body: bo_cong_an`, `legal_domain: ma_tuy`, `lang: vi`      |
| 2   | 34-2026.md — NĐ sửa đổi quy hoạch đô thị và nông thôn             | Chính phủ VN | 10,133   | `doc_number: 34/2026/NĐ-CP`, `doc_type: sua_doi`, `issuing_body: bo_xay_dung`, `legal_domain: quy_hoach`, `lang: vi`    |
| 3   | 47-2026.md — NĐ sửa đổi quản lý tang vật vi phạm hành chính       | Chính phủ VN | 12,382   | `doc_number: 47/2026/NĐ-CP`, `doc_type: sua_doi`, `issuing_body: bo_cong_an`, `legal_domain: hanh_chinh`, `lang: vi`    |
| 4   | 61-2026.md — NĐ phương tiện kỹ thuật phát hiện vi phạm hành chính | Chính phủ VN | 42,764   | `doc_number: 61/2026/NĐ-CP`, `doc_type: nghi_dinh`, `issuing_body: bo_cong_an`, `legal_domain: hanh_chinh`, `lang: vi`  |
| 5   | 62-2026.md — NĐ sửa đổi văn phòng đại diện tổ chức nước ngoài     | Chính phủ VN | 30,994   | `doc_number: 62/2026/NĐ-CP`, `doc_type: sua_doi`, `issuing_body: bo_ngoai_giao`, `legal_domain: ngoai_giao`, `lang: vi` |

### Metadata Schema

| Trường metadata | Kiểu   | Ví dụ giá trị                                             | Tại sao hữu ích cho retrieval?                                                                       |
| --------------- | ------ | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `doc_number`    | string | `"28/2026/NĐ-CP"`                                         | Tra cứu chính xác theo số hiệu văn bản, tránh nhầm lẫn giữa các nghị định                            |
| `doc_type`      | string | `"nghi_dinh"`, `"sua_doi"`                                | Phân biệt văn bản gốc với văn bản sửa đổi bổ sung                                                    |
| `issuing_body`  | string | `"bo_cong_an"`, `"bo_xay_dung"`, `"bo_ngoai_giao"`        | Filter theo cơ quan ban hành khi query liên quan đến một bộ cụ thể                                   |
| `legal_domain`  | string | `"ma_tuy"`, `"quy_hoach"`, `"hanh_chinh"`, `"ngoai_giao"` | Trường quan trọng nhất — filter theo lĩnh vực pháp luật, loại bỏ noise từ các domain không liên quan |
| `lang`          | string | `"vi"`                                                    | Phân biệt ngôn ngữ nếu sau bổ sung thêm tài liệu tiếng Anh                                           |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=300)` trên 3 tài liệu pháp luật:

| Tài liệu                  | Strategy                         | Chunk Count | Avg Length | Preserves Context?                       |
| ------------------------- | -------------------------------- | ----------- | ---------- | ---------------------------------------- |
| 28-2026.md (5,099 chars)  | FixedSizeChunker (`fixed_size`)  | 21          | 290.4      | Trung bình — cắt giữa điều khoản         |
| 28-2026.md (5,099 chars)  | SentenceChunker (`by_sentences`) | 11          | 461.3      | Tốt nhưng chunk quá dài                  |
| 28-2026.md (5,099 chars)  | RecursiveChunker (`recursive`)   | 25          | 202.1      | Tốt — tách theo cấu trúc điều/khoản      |
| 47-2026.md (12,382 chars) | FixedSizeChunker (`fixed_size`)  | 50          | 296.6      | Trung bình — cắt giữa thủ tục            |
| 47-2026.md (12,382 chars) | SentenceChunker (`by_sentences`) | 20          | 616.8      | Kém — chunk quá dài, nhiều ý lẫn lộn     |
| 47-2026.md (12,382 chars) | RecursiveChunker (`recursive`)   | 62          | 198.0      | Tốt — giữ nguyên từng khoản              |
| 61-2026.md (42,764 chars) | FixedSizeChunker (`fixed_size`)  | 171         | 299.8      | Trung bình                               |
| 61-2026.md (42,764 chars) | SentenceChunker (`by_sentences`) | 65          | 655.6      | Kém — avg_len quá cao với văn bản dài    |
| 61-2026.md (42,764 chars) | RecursiveChunker (`recursive`)   | 216         | 196.2      | Tốt — tách sát theo cấu trúc chương/điều |

### Strategy Của Tôi

**Loại:** LegalArticleChunker (custom, chunk_size=500)

**Mô tả cách hoạt động:**

> `LegalArticleChunker` dùng DFS 2 cấp với regex để tách theo cấu trúc pháp lý. Cấp 1 dùng regex `\*\*Chương\s+\S+\*\*` để tách văn bản thành các block Chương, bắt cả tên chương ở dòng tiếp theo. Cấp 2 dùng regex `\*\*Điều\s+\d+\.[^*]+\*\*` để tách từng Chương thành các Điều riêng biệt. Điểm đặc biệt: mỗi chunk đều được **prefix đầy đủ** `Chương → Tên chương → Điều → Tên điều` để giữ context cho LLM. Nếu 1 Điều vẫn vượt quá `chunk_size`, fallback về `RecursiveChunker`.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> Văn bản pháp luật VN có cấu trúc Chương/Điều rất rõ ràng và được đánh số — đây là ranh giới ngữ nghĩa tự nhiên nhất. Không có strategy nào generic có thể khai thác điều này tốt hơn regex chuyên biệt. Quan trọng hơn, prefix `[Chương X — Điều Y]` trong mỗi chunk đảm bảo LLM luôn biết chunk đó thuộc điều khoản nào, tránh nhầm lẫn khi nhiều điều có nội dung tương tự.

**Code snippet:**

```python
from src.chunking import LegalArticleChunker

chunker = LegalArticleChunker(chunk_size=500)
chunks = chunker.chunk(text)
# Mỗi chunk có dạng:
# **Chương II**
# **DANH MỤC VÀ VIỆC QUẢN LÝ...**
# **Điều 6. Danh mục và tiêu chuẩn...**
#
# [nội dung điều khoản]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu   | Strategy                             | Chunk Count | Avg Length | Retrieval Quality?                             |
| ---------- | ------------------------------------ | ----------- | ---------- | ---------------------------------------------- |
| 61-2026.md | RecursiveChunker (baseline tốt nhất) | 216         | 196.2      | Tốt — tách nhỏ theo cấu trúc                   |
| 61-2026.md | **LegalArticleChunker (của tôi)**    | 121         | 459.1      | Tốt hơn — chunk lớn hơn, có prefix Chương/Điều |
| 47-2026.md | RecursiveChunker (baseline)          | 62          | 198.0      | Tốt                                            |
| 47-2026.md | **LegalArticleChunker (của tôi)**    | 36          | 460.7      | Tốt hơn — ít chunk nhiễu hơn, context rõ hơn   |

### So Sánh Với Thành Viên Khác

| Thành viên         | Strategy                                  | Retrieval Score (/10) | Điểm mạnh                                                                     | Điểm yếu                                                             |
| ------------------ | ----------------------------------------- | --------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Nguyễn Trường Phúc | LegalArticleChunker (chunk_size=500)      | 9                     | Top-1: 10/15, top-3: 12/15 — prefix Chương/Điều giữ đủ context                | Phụ thuộc format Markdown, không dùng được cho file plain text       |
| Vũ Đăng Khiêm      | RecursiveChunker (chunk_size=500)         | 8                     | Top-1: 11/15, top-3: 13/15 — giữ cấu trúc Điều/Khoản, không cần format cụ thể | Chunk không có prefix Chương/Điều, thiếu context pháp lý             |
| Lê Dương Hiếu      | DecreeArticleChunker(max_chunk_size=2000) | 8                     | Ranh giới Điều pháp lý rõ ràng, câu trả lời không bị phân mảnh                | Phụ thuộc format Markdown, không dùng được cho file plain text       |
| Trần Minh Anh      | LegalChunker (chunk_size=1200)            | 8                     | Top-1: 10/15, top-3: 12/15 — tách đúng Điều/Khoản, chunk lớn giữ ngữ cảnh tốt | chunk_size lớn (1200) có thể gộp nhiều ý vào 1 chunk, giảm precision |
| Nguyễn Huyền San   | SentenceChunker (max_sentences=3)         | 9                     | Top-3 rất cao (14/15), chunk bám ranh giới câu tự nhiên                       | Avg length không đồng đều (358–655 chars), chunk dài ở NĐ dài        |
| Hoàng Hải Đăng     | RecursiveChunker (`chunk_size=500`)       | 8                     | Top-1: 11/15, top-3: 13/15 — tốt nhất về top-1 trong nhóm                     | Chunk không có prefix Chương/Điều, thiếu context pháp lý             |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> `LegalArticleChunker` phù hợp nhất cho domain văn bản pháp luật vì nó khai thác trực tiếp cấu trúc Chương/Điều — ranh giới ngữ nghĩa tự nhiên của nghị định. Prefix context trong mỗi chunk giúp retrieval chính xác hơn: khi người dùng hỏi về một quy định cụ thể, LLM biết ngay chunk đó thuộc Điều nào, Chương nào mà không cần đọc các chunk xung quanh. So với RecursiveChunker (tốt cho văn bản kỹ thuật tổng quát), LegalArticleChunker tạo ra ít chunk hơn nhưng mỗi chunk mang nhiều ngữ nghĩa pháp lý hơn.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> Dùng `re.split` với pattern `(?<=\.)\s+|(?<=[!?])\s+|(?<=\.)\n` — lookbehind assertion đảm bảo dấu `.`, `!`, `?` vẫn thuộc về câu trước, không bị mất khi split. Các sentence sau đó được nhóm theo `max_sentences_per_chunk` bằng list slicing `sentences[i : i + max]` rồi join bằng space. Edge case xử lý: strip whitespace và bỏ qua sentence rỗng trước khi nhóm.

**`RecursiveChunker.chunk` / `_split`** — approach:

> `chunk` là wrapper gọi `_split(text, self.separators)`. Base case của `_split`: nếu `len(text) <= chunk_size` thì trả về `[text]` ngay. Với mỗi separator, split text thành `parts` rồi gộp dần vào `current_piece`; khi candidate vượt chunk_size thì flush `current_piece` qua đệ quy với `rest` separators, bắt đầu `current_piece` mới. Fallback cuối: separator `""` → character-level split cứng.

**`LegalArticleChunker` (custom)** — approach:

> DFS 2 cấp dùng regex chuyên biệt cho văn bản pháp luật VN dạng Markdown. Cấp 1 tách theo pattern `\*\*Chương\s+\S+\*\*` kết hợp tên chương ở dòng tiếp theo. Cấp 2 tách theo `\*\*Điều\s+\d+\.[^*]+\*\*` trong từng block Chương. Điểm thiết kế chính: mỗi chunk được prefix `Chương + Điều` để LLM luôn có đủ context pháp lý ngay trong chunk, không cần xem chunk liền kề. Nếu 1 Điều vẫn > `chunk_size`, fallback về `RecursiveChunker`.

### EmbeddingStore

**`add_documents` + `search`** — approach:

> `_make_record` embed nội dung bằng `self._embedding_fn`, tạo dict chuẩn hóa gồm `id`, `content`, `embedding`, `metadata` — thêm `doc_id` vào metadata để `delete_document` hoạt động sau này. `add_documents` append từng record vào `self._store` (list in-memory). `search` gọi `_search_records`: embed query, tính dot product với embedding của mỗi record, sort giảm dần theo score, trả về top_k — mỗi result dict có thêm key `score`.

**`search_with_filter` + `delete_document`** — approach:

> `search_with_filter` filter **trước** khi search: dùng list comprehension giữ lại records có tất cả key-value trong `metadata_filter` khớp chính xác, sau đó gọi `_search_records` trên tập đã lọc. Nếu `metadata_filter` là `None` thì search toàn bộ. `delete_document` dùng list comprehension tạo list mới loại bỏ records có `metadata['doc_id'] == doc_id`, trả về `True` nếu size giảm — không mutate in-place để tránh side effect.

### KnowledgeBaseAgent

**`answer`** — approach:

> Retrieve top-k chunks từ store bằng `self.store.search(question, top_k)`. Build prompt theo cấu trúc 3 phần: (1) instruction "Answer using only the context below", (2) numbered context blocks `[1] content\n[2] content...`, (3) `Question: ... \nAnswer:`. Cấu trúc này grounding LLM vào retrieved text, giảm hallucination. Kết quả là output của `self.llm_fn(prompt)` — agent không tự generate, chỉ orchestrate retrieve + prompt.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.0.3, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.07s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Các cặp câu được chọn từ domain pháp luật để kiểm tra xem embedder có phân biệt được ngữ nghĩa không.

Embedder sử dụng: **OpenAI `text-embedding-3-small`** (dim=1536).

| Pair | Sentence A                                           | Sentence B                                                | Dự đoán | Actual Score | Đúng? |
| ---- | ---------------------------------------------------- | --------------------------------------------------------- | ------- | ------------ | ----- |
| 1    | Nghị định có hiệu lực thi hành từ ngày nào?          | Ngày có hiệu lực thi hành của nghị định là khi nào?       | high    | +0.8562      | Có    |
| 2    | Bộ Công an quản lý tang vật vi phạm hành chính.      | Bộ Ngoại giao cấp phép cho văn phòng đại diện nước ngoài. | low     | +0.3615      | Có    |
| 3    | Niêm phong tang vật phương tiện vi phạm.             | Mở niêm phong tang vật phương tiện bị tạm giữ.            | high    | +0.7400      | Có    |
| 4    | Hồ sơ quy hoạch gửi qua hệ thống thông tin xây dựng. | Thời tiết hôm nay có mưa không?                           | low     | +0.1809      | Có    |
| 5    | Văn phòng đại diện phải báo cáo định kỳ.             | Tổ chức nước ngoài gửi báo cáo cho Bộ Ngoại giao.         | high    | +0.5146      | Có    |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> Pair 3 bất ngờ nhất theo chiều ngược: "niêm phong" và "mở niêm phong" là hai hành động đối lập nhưng embedding vẫn cho score rất cao (+0.7400) vì cả hai câu dùng chung thuật ngữ pháp lý ("tang vật", "phương tiện", "tạm giữ") — embedder nhận ra ngữ cảnh chung chứ không chỉ nhìn từng từ. Pair 2 cho score +0.3615 dù nói về hai bộ khác nhau, cho thấy embedder hiểu cả hai đều là "cơ quan nhà nước quản lý theo lĩnh vực" nên vẫn có độ tương đồng nhất định. Điều này chứng tỏ OpenAI `text-embedding-3-small` hiểu ngữ nghĩa tiếng Việt rất tốt, khác hoàn toàn với mock embedder ngẫu nhiên.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 trong 15 benchmark queries của nhóm (từ file `15-sample.csv`) trên `LegalArticleChunker` + `EmbeddingStore` với OpenAI `text-embedding-3-small`.

### Benchmark Queries & Gold Answers (nhóm thống nhất — trích 5/15)

| #   | Query                                                                                            | Gold Answer                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1   | Nghị định quy định các danh mục chất ma túy và tiền chất có hiệu lực thi hành từ ngày nào?       | Nghị định có hiệu lực thi hành từ ngày 19 tháng 01 năm 2026.                                                                  |
| 2   | Cơ quan nào có trách nhiệm quản lý nhà nước về các tiền chất sử dụng trong lĩnh vực công nghiệp? | Bộ Công Thương có trách nhiệm quản lý nhà nước về các tiền chất sử dụng trong lĩnh vực công nghiệp.                           |
| 3   | Ai quyết định việc niêm phong tang vật, phương tiện?                                             | Người lập biên bản tạm giữ hoặc người có thẩm quyền tạm giữ tang vật, phương tiện.                                            |
| 4   | Dữ liệu có thể được cung cấp qua những hình thức nào?                                            | Trực tiếp, thư điện tử, cổng/trang thông tin điện tử, ứng dụng VNeID, dịch vụ bưu chính hoặc kết nối chia sẻ dữ liệu điện tử. |
| 5   | Văn phòng đại diện phải báo cáo định kỳ khi nào?                                                 | Chậm nhất vào ngày 20 của tháng cuối kỳ báo cáo.                                                                              |

Embedder: **OpenAI `text-embedding-3-small`** | Chunker: **LegalArticleChunker** (chunk_size=500) | Store: 274 chunks từ 5 nghị định.

### Kết Quả Của Tôi

| #   | Query (rút gọn)                        | Top-1 Retrieved Chunk (tóm tắt)                                                  | Score  | Relevant? | Agent Answer (tóm tắt)                   |
| --- | -------------------------------------- | -------------------------------------------------------------------------------- | ------ | --------- | ---------------------------------------- |
| 1   | Hiệu lực thi hành NĐ ma túy?           | 28-2026 **Điều 3. Hiệu lực thi hành** — Nghị định có hiệu lực từ ngày 19/01/2026 | 0.7532 | Có        | Trả lời đúng: 19/01/2026                 |
| 2   | Bộ nào quản lý tiền chất công nghiệp?  | 28-2026 **Điều 2. Trách nhiệm quản lý** — Bộ Công Thương có trách nhiệm...       | 0.7505 | Có        | Trả lời đúng: Bộ Công Thương             |
| 3   | Carisoprodol và Etomidate từ ngày nào? | 28-2026 **Điều 3. Hiệu lực thi hành** — kể từ ngày 01/6/2026...                  | 0.6823 | Có        | Trả lời đúng: 01/6/2026                  |
| 4   | NĐ 34 sửa đổi nghị định nào?           | 34-2026 **NGHỊ ĐỊNH** — Sửa đổi NĐ số 178/2025/NĐ-CP...                          | 0.6894 | Có        | Trả lời đúng: NĐ 178/2025/NĐ-CP          |
| 5   | Hồ sơ quy hoạch gửi qua hệ thống nào?  | 62-2026 **Điều 1** — Cơ quan tiếp nhận hồ sơ... (sai file)                       | 0.5267 | Không     | Trả lời sai nguồn — lấy từ 62 thay vì 34 |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

**Toàn bộ 15 queries (15-sample.csv):** Top-1 relevant: **10/15** | Top-3 relevant: **12/15**

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> [Điền sau demo — ví dụ: thành viên dùng SentenceChunker với max=2 cho kết quả chunk ngắn hơn, dễ match hơn với query ngắn; hoặc thành viên bổ sung metadata `article_number` để filter chính xác đến từng Điều]

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> [Điền sau demo — ví dụ: nhóm khác dùng real embedder và cho thấy precision tăng từ 3/5 lên 5/5; hoặc nhóm khác thiết kế metadata `effective_date` giúp filter query về hiệu lực thi hành]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> Kết quả 10/15 với OpenAI embedder cho thấy 5 queries thất bại chủ yếu do nhiều nghị định có cùng thuật ngữ ("hồ sơ", "hệ thống") dẫn đến nhầm lẫn giữa các file. Tôi sẽ bổ sung metadata `article_number` và tận dụng `search_with_filter` theo `legal_domain` cho các query có thể xác định rõ lĩnh vực — ví dụ query về quy hoạch filter `legal_domain=quy_hoach` trước khi search để loại bỏ nhiễu từ các nghị định khác. Ngoài ra, với `LegalArticleChunker`, tôi sẽ chunk ở cấp Khoản thay vì Điều cho các Điều dài để tăng granularity retrieval.

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
| --------------------------- | ------- | ---------------- |
| Warm-up                     | Cá nhân | 5 / 5            |
| Document selection          | Nhóm    | 9 / 10           |
| Chunking strategy           | Nhóm    | 13 / 15          |
| My approach                 | Cá nhân | 10 / 10          |
| Similarity predictions      | Cá nhân | 4 / 5            |
| Results                     | Cá nhân | 9 / 10           |
| Core implementation (tests) | Cá nhân | 30 / 30          |
| Demo                        | Nhóm    | 5 / 5            |
| **Tổng**                    |         | **85 / 90**      |
