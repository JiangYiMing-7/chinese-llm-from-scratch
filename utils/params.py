def count_parameters(model) -> int:
    # 统计参数总量，注意 tie_embeddings 时输入输出 embedding 共享，这里只算一次
    return sum(p.numel() for p in model.parameters())
