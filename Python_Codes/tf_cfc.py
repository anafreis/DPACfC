import tensorflow as tf
import numpy as np

def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(0.666 * x)


class CfcCell(tf.keras.layers.Layer):
    def __init__(self, units, hparams, **kwargs):
        super(CfcCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.hparams = hparams
        self._no_gate = False
        self.cfc_name = kwargs['name']

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        if self.hparams.get("backbone_activation") == "relu":
            backbone_activation = tf.nn.relu
        elif self.hparams.get("backbone_activation") == "gelu":
            backbone_activation = tf.nn.gelu
        else:
            raise ValueError("Unknown backbone activation")
   
        self.concat = tf.keras.layers.Concatenate(name = self.cfc_name+'_concat')

        self.backbone = []
        for i in range(self.hparams["backbone_layers"]):

            self.backbone.append(
                tf.keras.layers.Dense(
                    self.hparams["backbone_units"],
                    backbone_activation,
                    kernel_regularizer=tf.keras.regularizers.L2(
                        self.hparams["weight_decay"]
                    ),
                    name = self.cfc_name+'_dense_backbone_'+str(i),
                )
            )
            self.backbone.append(tf.keras.layers.Dropout(self.hparams["backbone_dr"], name = self.cfc_name+'_dropout_backbone_'+str(i)))

        self.backbone = tf.keras.models.Sequential(self.backbone)
        
        self.ff1 = tf.keras.layers.Dense(
            self.units,
            lecun_tanh,
            kernel_regularizer=tf.keras.regularizers.L2(
                self.hparams["weight_decay"]
            ),
            name = self.cfc_name+'_dense_ff1',
        )
        self.ff2 = tf.keras.layers.Dense(
            self.units,
            lecun_tanh,
            kernel_regularizer=tf.keras.regularizers.L2(
                self.hparams["weight_decay"]
            ),
            name = self.cfc_name+'_dense_ff2',
        )
        self.time_a = tf.keras.layers.Dense(
            self.units,
            kernel_regularizer=tf.keras.regularizers.L2(
                self.hparams["weight_decay"]
            ),
            name = self.cfc_name+'_dense_time_a',
        )
        self.time_b = tf.keras.layers.Dense(
            self.units,
            kernel_regularizer=tf.keras.regularizers.L2(
                self.hparams["weight_decay"]
            ),
            name = self.cfc_name+'_dense_time_b',
        )
        self.built = True

    def call(self, inputs, states, **kwargs):
        hidden_state = states[0]
        t = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            t = tf.reshape(elapsed, [-1, 1])
            inputs = inputs[0]

        x = self.concat([inputs, hidden_state])
        x = self.backbone(x)
        ff1 = self.ff1(x)
        # Cfc
        ff2 = self.ff2(x)
        t_a = self.time_a(x)
        t_b = self.time_b(x)
        t_interp = tf.nn.sigmoid(-t_a * t + t_b)
        new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, [new_hidden]