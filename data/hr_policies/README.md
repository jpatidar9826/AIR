# HR policy corpus for the RAG module

The `.md` files in this folder are **sample policies written for this project**
(not real TCS or company policies) so the RAG demo works out of the box.

To use a public corpus instead, drop `.pdf`, `.docx`, `.txt` or `.md` files
here (for example, publicly available employee handbooks) and rebuild the index:

    python run_pipeline.py --steps rag

Never place confidential or internal documents in this folder.
