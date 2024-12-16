# Классификатор спектров

```elixir
Mix.install(
  [
    {:axon, "~> 0.7"},
    {:nx, "~> 0.9"},
    {:exla, "~> 0.9"},
    {:kino, "~> 0.14.2"},
    {:scholar, "~> 0.3.0"}
  ],
  config: [
    nx: [
      default_backend: EXLA.Backend,
      default_defn_options: [compiler: EXLA, lazy_transfers: :always]
    ],
    exla: [
      default_client: :cuda,
      clients: [
        host: [platform: :host],
        cuda: [platform: :cuda]
      ]
    ]
  ],
  system_env: [
    XLA_TARGET: "cuda12"
  ]
)


Nx.default_backend(EXLA.Backend)

Application.put_env(:exla, :clients,
  cuda: [platform: :cuda]
)


```

## Парсинг файлов

```elixir
defmodule Thermosol.SpectrumFileParser do
  def fetch_data_from_files(path) do
    files = Path.wildcard(path)

    labels = fetch_labels(files)

    data =
      files
      |> Task.async_stream(&read_file/1)
      |> Stream.map(fn {:ok, {:ok, data}} -> data end)

    {data, labels}
  end

  # Data

  defp read_file(path) do
    case process_file(path) do
      [] -> {:error, :empty_file}
      data -> {:ok, data}
    end
  rescue
    _ in ArgumentError ->
      {:error, :invalid_data}
  end

  defp process_file(file) do
    file
    |> File.stream!()
    |> Enum.drop(1)
    |> Enum.map(&process_line/1)
    |> Enum.filter(fn [wn, _] -> wn >= 100 end)
  end

  defp process_line(line) do
    [wn, od] = String.split(line)
    [parse_to_int(wn), String.to_float(od)]
  end

  defp parse_to_int(string) do
    string
    |> String.to_float()
    |> trunc()
  end

  # Labels

  defp fetch_labels(files) do
    files
    |> Stream.map(&extract_mineral_name/1)
  end

  defp extract_mineral_name(path) do
    case String.split(path, "/") do
      splitted_path -> Enum.at(splitted_path, -2)
    end
  end
end

```

## Модуль подготовки данных

```elixir
defmodule Thermosol.SpectrumDataPreparing do
  import Nx
  import Nx.Defn

  alias Scholar.Interpolation.Linear, as: Interpolation
  alias Scholar.Preprocessing.MinMaxScaler, as: MinMaxScaler

  # Preparing pipelines

  def train_test_pipelines(data, labels, batch_size \\ 15, test_size \\ 30, val_size \\ 5) do
    data =
      data
      |> Enum.map(&Nx.tensor/1)
      |> Enum.map(&process_spectrum/1)

    {data, labels} =
      Enum.zip(data, labels)
      |> Enum.shuffle()
      |> Enum.unzip()

    {encoded_labels, one_hot_dictionary} = encode_labels(labels)

    full_dataset = Enum.zip(data, encoded_labels)

    {test_dataset, train_dataset} =
      full_dataset
      |> Enum.split(test_size)

    {val_dataset, train_dataset} =
      train_dataset
      |> Enum.split(val_size)

    {test_data, test_labels} = Enum.unzip(test_dataset)
    {train_data, train_labels} = Enum.unzip(train_dataset)
    {val_data, val_labels} = Enum.unzip(val_dataset)

    {
      pipeline(test_data, test_labels, batch_size),
      pipeline(train_data, train_labels, batch_size),
      pipeline(val_data, val_labels, batch_size),
      one_hot_dictionary
    }
  end

  defp pipeline(data, labels, batch_size) do
    data
    |> Stream.zip(labels)
    |> Stream.chunk_every(batch_size, batch_size, :discard)
    |> Stream.map(fn chunks ->
      {spectrum_chunk, label_chunk} = Enum.unzip(chunks)
      {Nx.stack(spectrum_chunk), Nx.stack(label_chunk)}
    end)
  end

  # Processing spectrum data

  defnp process_spectrum(%Nx.Tensor{} = spectrum) do
    spectrum
    |> transpose()
    |> interpolate()
    |> fetch_intensity_row()
    |> min_max_scale()
    |> build_bitmap()
  end

  defnp interpolate(spectrum) do
    target_x =
      linspace(400, 3999, n: 900)
      |> Nx.new_axis(0)

    target_y =
      Interpolation.fit(spectrum[0], spectrum[1])
      |> Interpolation.predict(target_x)

    Nx.concatenate([target_x, target_y])
  end

  defnp fetch_intensity_row(spectrum) do
    spectrum[1]
  end

  defnp build_bitmap(intensity_row) do
    scale = 100
    shape = {scale, size(intensity_row)}

    intensity_row =
      intensity_row
      |> Nx.multiply(scale)
      |> Nx.round()

    greater_equal(
      iota(shape, axis: 0),
      intensity_row
    )
    |> new_axis(-1)
    |> multiply(255)
    |> as_type({:u, 8})
  end

  defnp min_max_scale(spectrum) do
    MinMaxScaler.fit_transform(spectrum)
  end

  # Processing spectrum labels

  def encode_labels(labels) do
    unique_labels = Enum.uniq(labels)
    one_hot_tensor_size = length(unique_labels)

    one_hot_dictionary =
      unique_labels
      |> Enum.with_index(fn name, index ->
        {name, Nx.equal(index, Nx.iota({one_hot_tensor_size}))}
      end)
      |> Enum.into(%{})

    encoded_labels =
      labels
      |> Stream.map(fn label -> one_hot_dictionary[label] end)

    {encoded_labels, one_hot_dictionary}
  end
end

train_path = "/home/hassan/projects/datasets/3_minerals/*/*.txt"

{data, labels} = Thermosol.SpectrumFileParser.fetch_data_from_files(train_path)

{train_pipeline, test_pipeline, val_pipeline, one_hot_dictionary} =
  Thermosol.SpectrumDataPreparing.train_test_pipelines(data, labels, 5, 20)

```

## Пример спектра

```elixir
[{a, _}] = Enum.take(val_pipeline, 1)
a[0]
|> Kino.Image.new()
```

[![Spectrum](./img/spectrum.png)](./img/spectrum.png)

## Модуль свёрточной модели

```elixir
defmodule Thermosol.SpectrumCNNModel do
  def create() do
    Axon.input("spectra", shape: {nil, 100, 900, 1})
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu, padding: :same)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2])
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu, padding: :same)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2])
    |> Axon.conv(128, kernel_size: {3, 3}, activation: :relu, padding: :same)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: [2, 2])
    |> Axon.flatten()
    |> Axon.dense(128, activation: :relu)
    |> Axon.dense(3, activation: :softmax)
  end

  def fit(model, train_pipeline, val_pipeline, epochs) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.validate(model, val_pipeline)
    |> Axon.Loop.early_stop("validation_loss", mode: :min)
    |> Axon.Loop.run(train_pipeline, %{}, epochs: epochs)
  end

  def test(model, model_state, test_pipeline) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(test_pipeline, model_state, compiler: EXLA)
  end
end

alias Thermosol.SpectrumCNNModel, as: CNN
model = CNN.create()
model_state = CNN.fit(model, train_pipeline, val_pipeline, 100)
CNN.test(model, model_state, test_pipeline)

```

```
09:00:18.043 [warning] passing parameter map to initialization is deprecated, use %Axon.ModelState{} instead
Epoch: 0, Batch: 0, accuracy: 0.6000000 loss: 0.0000000
Epoch: 1, Batch: 0, accuracy: 0.2000000 loss: 1654.8771973
Epoch: 2, Batch: 0, accuracy: 0.6000000 loss: 886.9898071
Epoch: 3, Batch: 0, accuracy: 0.8000000 loss: 592.4161377
Epoch: 4, Batch: 0, accuracy: 1.0000000 loss: 444.7496643
Epoch: 5, Batch: 0, accuracy: 1.0000000 loss: 355.8287659
Batch: 0, accuracy: 0.8000000 loss: 1.4684908
09:00:23.475 [debug] Forwarding options: [compiler: EXLA] to JIT compiler
Batch: 27, accuracy: 0.6714288

%{
  0 => %{
    "accuracy" => #Nx.Tensor<
      f32
      EXLA.Backend<cuda:0, 0.2401505803.2796159043.11421>
      0.6714287996292114
    >
  }
}
```

## Что дальше?

1. Реализовать аугментацию данных. Данных мало - всего 150 спектров и 3 класса.
2. Реализовать регуляризацию.
3. Провести кросс-валидацию для гиперпараметров сверточной модели.
4. Отрефакторить код.
5. Прикрутить модуль к проекту.
