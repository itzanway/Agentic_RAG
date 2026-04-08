def sliding_window_chunker(text_blocks, chunk_size=500, overlap=50):
    """
    Splits text into chunks of roughly `chunk_size` words with an overlap.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for block in text_blocks:
        words = block.split()
        for word in words:
            current_chunk.append(word)
            current_length += 1

            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep the overlap for the next chunk
                current_chunk = current_chunk[-overlap:]
                current_length = len(current_chunk)
                
    if current_chunk:
         chunks.append(" ".join(current_chunk))
         
    return chunks