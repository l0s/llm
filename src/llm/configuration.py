class Configuration:
    def __init__(
        self,
        vocabulary_size: int = 50_257,
        context_length: int = 1_024,
        embedding_dimension: int = 768,
        head_count: int = 12,
        layer_count: int = 12,
        drop_rate: float = 0.1,
        embedding_dropout_rate: float | None = None,
        shortcut_dropout_rate: float | None = None,
        attention_dropout_rate: float | None = None,
        bias: bool = False,
    ):
        """
        :param vocabulary_size: the number of tokens
        :param context_length: the maximum number of tokens the model can process at-a-time with positional embeddings
        :param embedding_dimension: the embedding size, the size of each vector that represents a token
        :param head_count: the number of attention heads
        :param layer_count: the number of transformer blocks
        :param drop_rate: the dropout intensity to prevent over-fitting
        :param embedding_dropout_rate: the dropout intensity for the embedding layer (defaults to `drop_rate`)
        :param shortcut_dropout_rate: the dropout intensity for the shortcut layer (defaults to `drop_rate`)
        :param attention_dropout_rate: the dropout intensity for the attention layer (defaults to `drop_rate`)
        :param bias: whether to include a bias vector for the queries, keys, and values
        """
        self.vocabulary_size = vocabulary_size
        self.context_length = context_length
        self.embedding_dimension = embedding_dimension
        self.head_count = head_count
        self.layer_count = layer_count
        self.drop_rate = drop_rate
        self.embedding_dropout_rate = embedding_dropout_rate or drop_rate
        self.shortcut_dropout_rate = shortcut_dropout_rate or drop_rate
        self.attention_dropout_rate = attention_dropout_rate or drop_rate
        self.bias = bias
