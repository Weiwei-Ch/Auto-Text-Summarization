import numpy as np
import tensorflow as tf
from src.pgn_transformer_tf2.layers.transformer import MultiHeadAttention
from src.pgn_transformer_tf2.layers.common import point_wise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # ------------------------------------------------------------------------------
        # 补全代码
        # 对 x 使用多头注意力机制 
        # 使用 dropout 层
        # 使用 layernorm 层
        # 使用 ffn
        # 使用 dropout2
        # 使用 layernorm 层
        # ------------------------------------------------------------------------------
        # x:(batch_size, input_seq_len, d_model)
        atten_output, _ = self.mha(x, x, x, mask)  #(batch_size, input_seq_len, d_model)
        atten_output = self.dropout1(atten_output, training=training)
        #残差
        out1 = self.layernorm1(x + atten_output)   
        
        ffn_output = self.ffn(out1)   # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        #残差        
        out2 = self.laynernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)
        
        return out2


if __name__ == '__main__':
    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    # (batch_size, input_seq_len, d_model)
    print(sample_encoder_layer_output.shape)


