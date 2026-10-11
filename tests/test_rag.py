from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from pypdf import PdfWriter

from rag import build_index, generate_answer, read_chunks, retrieve


class RagTests(unittest.TestCase):
    def test_real_pdf_text_extraction(self):
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=400, height=400)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 30 300 Td (Attendance is 75 percent.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        data = BytesIO()
        writer.write(data)
        chunks, skipped = read_chunks(data.getvalue())
        self.assertEqual(skipped, 0)
        self.assertEqual(chunks[0], {"text": "Attendance is 75 percent.", "page": 1})

    def test_blank_pdf_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        data = BytesIO()
        writer.write(data)
        with self.assertRaisesRegex(ValueError, "No readable text"):
            read_chunks(data.getvalue())

    @patch("rag.PdfReader")
    def test_page_numbers_and_chunk_boundaries(self, reader):
        reader.return_value.is_encrypted = False
        reader.return_value.pages = [
            Mock(extract_text=Mock(return_value="x" * 350)),
            Mock(extract_text=Mock(return_value="")),
            Mock(extract_text=Mock(return_value="Library opens at nine.")),
        ]
        chunks, skipped = read_chunks(b"pdf", size=180, overlap=30)
        self.assertEqual(skipped, 1)
        self.assertEqual([c["page"] for c in chunks], [1, 1, 1, 3])
        self.assertEqual(len(chunks[2]["text"]), 50)
        self.assertTrue(all(len(c["text"]) <= 180 for c in chunks))

    @patch("rag.read_chunks")
    def test_chroma_retrieval_preserves_source(self, read):
        read.return_value = ([{"text": "Attendance is 75%.", "page": 2},
                               {"text": "Library opens at nine.", "page": 5}], 0)
        model = Mock()
        model.encode.side_effect = [[[1., 0.], [0., 1.]], [[0., 1.]]]
        client, collection, _, _ = build_index(b"pdf", model)
        try:
            matches = retrieve(collection, model, "library hours")
            self.assertEqual(matches[0], {"text": "Library opens at nine.", "page": 5})
            self.assertEqual(len(matches), 2)
        finally:
            client.delete_collection(collection.name)

    @patch("rag.requests.post")
    def test_llm_receives_question_and_page_context(self, post):
        post.return_value.json.return_value = {"message": {"content": "75% [Page 2]"}}
        answer = generate_answer("Attendance?", [{"text": "75%", "page": 2}],
                                 "http://localhost:11434/", "test-model")
        self.assertEqual(answer, "75% [Page 2]")
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/chat")
        payload = post.call_args.kwargs["json"]
        self.assertIn("[Page 2]", payload["messages"][1]["content"])
        self.assertIn("Attendance?", payload["messages"][1]["content"])
        self.assertFalse(payload["think"])


if __name__ == "__main__":
    unittest.main()
