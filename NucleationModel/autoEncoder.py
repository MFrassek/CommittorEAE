from tensorflow import keras


class AutoEncoder:
    @staticmethod
    def make_models(dimensions, const):
        encoder = AutoEncoder.make_encoder(
            const=const,
            dimensions=dimensions)

        decoder_1 = AutoEncoder.make_decoder(
            const=const,
            dimensions=dimensions,
            hidden_activation_function=const.decoder_1_act_func,
            hidden_layer_cnt=const.decoder_1_hidden,
            output_units=1,
            output_activation_function=const.decoder_1_act_func,
            output_name=const.output_name_1)

        decoder_2 = AutoEncoder.make_decoder(
            const=const,
            dimensions=dimensions,
            hidden_activation_function=const.decoder_2_act_func,
            hidden_layer_cnt=const.decoder_2_hidden,
            output_units=dimensions,
            output_activation_function=None,
            output_name=const.output_name_2)

        autoencoder = AutoEncoder.make_autoencoder(
            const=const,
            dimensions=dimensions,
            encoder=encoder,
            decoders=[decoder_1, decoder_2],
            name="Autoencoder")

        autoencoder_1 = AutoEncoder.make_autoencoder(
            const=const,
            dimensions=dimensions,
            encoder=encoder,
            decoders=[decoder_1],
            name="Autoencoder_1")

        autoencoder_2 = AutoEncoder.make_autoencoder(
            const=const,
            dimensions=dimensions,
            encoder=encoder,
            decoders=[decoder_2],
            name="Autoencoder_2")

        autoencoder.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss={
                const.output_name_1: const.loss_function_1,
                const.output_name_2: const.loss_function_2},
            loss_weights=[
                const.label_loss_weight,
                const.reconstruction_loss_weight])

        autoencoder_1.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss={const.output_name_1: const.loss_function_1},
            loss_weights=[const.label_loss_weight])

        autoencoder_2.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss={const.output_name_2: const.loss_function_2},
            loss_weights=[const.reconstruction_loss_weight])

        return autoencoder, autoencoder_1, autoencoder_2, \
            encoder, decoder_1, decoder_2

    @staticmethod
    def make_encoder(
            const,
            dimensions):
        encoder_input = keras.Input(
            shape=(dimensions,),
            name=const.input_name)
        x = encoder_input
        for i in range(const.encoder_hidden):
            x = keras.layers.Dense(
                units=dimensions * const.node_mult,
                activation=const.encoder_act_func)(x)
        encoder_output = keras.layers.Dense(
            units=const.bottleneck_size,
            activation=const.encoder_act_func,
            name="Bottleneck")(x)
        encoder = keras.Model(
            encoder_input,
            encoder_output,
            name="Encoder")
        return encoder

    @staticmethod
    def make_decoder(
            const,
            dimensions,
            hidden_layer_cnt,
            hidden_activation_function,
            output_units,
            output_activation_function,
            output_name):
        decoder_input = keras.Input(
            shape=(const.bottleneck_size,),
            name="Encoded")
        x = decoder_input
        for i in range(hidden_layer_cnt):
            x = keras.layers.Dense(
                units=dimensions * const.node_mult,
                activation=hidden_activation_function)(x)
        decoder_output = keras.layers.Dense(
            units=output_units,
            activation=output_activation_function,
            name=output_name)(x)
        decoder = keras.Model(
            decoder_input,
            decoder_output,
            name=output_name)
        return decoder

    @staticmethod
    def make_autoencoder(
            const,
            dimensions,
            encoder,
            decoders,
            name):
        autoencoder_input = keras.Input(
            shape=(dimensions,),
            name=const.input_name)
        encoded_snaphot = encoder(autoencoder_input)
        autoencoder_outputs = [decoder(encoded_snaphot)
                               for decoder in decoders]
        autoencoder = keras.Model(
            inputs=autoencoder_input,
            outputs=autoencoder_outputs,
            name=name)
        return autoencoder

    @staticmethod
    def visualize(model, file_name: str):
        keras.utils.plot_model(
            model,
            file_name,
            show_shapes=True)
