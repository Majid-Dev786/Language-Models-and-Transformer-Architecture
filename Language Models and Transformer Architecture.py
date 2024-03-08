import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the Transformer architecture
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=True):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.layer_norm1(inputs + attention_output)
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layer_norm2(output1 + ffn_output)
        return output2

# Create the language model
class LanguageModel(keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, feed_forward_dim, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(LanguageModel, self).__init__()
        self.embedding = layers.Embedding(input_vocab_size, embed_dim)
        self.pos_encoding = layers.Embedding(maximum_position_encoding, embed_dim)
        self.enc_layers = [TransformerEncoder(embed_dim, num_heads, feed_forward_dim, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=True):
        seq_length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = self.pos_encoding(positions)
        x = self.embedding(inputs)
        x += positions
        x = self.dropout(x, training=training)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training)
        return x

# Define hyperparameters and create the language model instance
num_layers = 2
embed_dim = 32
num_heads = 4
feed_forward_dim = 64
input_vocab_size = 10000
maximum_position_encoding = 1000
dropout_rate = 0.1

language_model = LanguageModel(num_layers, embed_dim, num_heads, feed_forward_dim,
                               input_vocab_size, maximum_position_encoding, dropout_rate)

# Test the language model
sequence_length = 20
sample_input = tf.random.uniform((64, sequence_length), maxval=input_vocab_size, dtype=tf.int64)
sample_output = language_model(sample_input, training=False)

print("Sample input shape:", sample_input.shape)
print("Sample output shape:", sample_output.shape)

