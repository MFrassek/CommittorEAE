from tensorflow import keras
import tensorflow as tf
import keras.backend as K


class AutoEncoder:
    @staticmethod
    def model(dimensions, loss_function, const):
        encoder_input = keras.Input(
            shape=(dimensions,),
            name=const.input_name)
        x = keras.layers.Dense(
            dimensions * const.node_mult,
            activation=const.encoder_act_func)(encoder_input)
        for i in range(const.encoder_hidden - 1):
            x = keras.layers.Dense(
                dimensions * const.node_mult,
                activation=const.encoder_act_func)(x)

        encoder_output = keras.layers.Dense(
            const.bottleneck_size,
            activation=const.encoder_act_func,
            name="bottleneck")(x)

        encoder = keras.Model(
            encoder_input,
            encoder_output,
            name="Encoder")

        decoder_input = keras.Input(
            shape=(const.bottleneck_size,),
            name="encoded_snapshots")

        x1 = keras.layers.Dense(
            dimensions * const.node_mult,
            activation=const.decoder_1_act_func)(decoder_input)
        for i in range(const.decoder_1_hidden):
            x1 = keras.layers.Dense(
                dimensions * const.node_mult,
                activation=const.decoder_1_act_func)(x1)

        decoder_output_1 = keras.layers.Dense(
            1,
            activation=const.decoder_1_act_func,
            name=const.output_name_1)(x1)

        decoder_1 = keras.Model(
            decoder_input,
            decoder_output_1,
            name=const.output_name_1)

        x2 = keras.layers.Dense(
            dimensions * const.node_mult,
            activation=const.decoder_2_act_func)(decoder_input)
        for i in range(const.decoder_2_hidden):
            x2 = keras.layers.Dense(
                dimensions * const.node_mult,
                activation=const.decoder_2_act_func)(x2)

        decoder_output_2 = keras.layers.Dense(
            dimensions,
            activation=const.decoder_2_act_func,
            name=const.output_name_2)(x2)

        decoder_2 = keras.Model(
            decoder_input,
            decoder_output_2,
            name=const.output_name_2)

        autoencoder_input = keras.Input(
            shape=(dimensions,),
            name=const.input_name)
        encoded_snaphot = encoder(autoencoder_input)
        label_snapshot = decoder_1(encoded_snaphot)
        reconstructed_snapshot = decoder_2(encoded_snaphot)

        autoencoder = keras.Model(
            inputs=autoencoder_input,
            outputs=[label_snapshot, reconstructed_snapshot],
            name="Autoencoder")

        autoencoder_1 = keras.Model(
            inputs=autoencoder_input,
            outputs=label_snapshot,
            name="Autoencoder_1")

        autoencoder_2 = keras.Model(
            inputs=autoencoder_input,
            outputs=reconstructed_snapshot,
            name="Autoencoder_2")

        autoencoder.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss={
                const.output_name_1: loss_function,
                const.output_name_2: keras.losses.MeanAbsoluteError()},
            loss_weights=[
                const.label_loss_weight,
                const.reconstruction_loss_weight])

        autoencoder_1.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss={const.output_name_1: loss_function},
            loss_weights=[const.label_loss_weight])

        autoencoder_2.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss={const.output_name_2: keras.losses.MeanAbsoluteError()},
            loss_weights=[const.reconstruction_loss_weight])
        return autoencoder, autoencoder_1, autoencoder_2, \
            encoder, decoder_1, decoder_2

    @staticmethod
    def visualize(model, file_name: str):
        model_layout = keras.utils.plot_model(
            model,
            file_name,
            show_shapes=True)
