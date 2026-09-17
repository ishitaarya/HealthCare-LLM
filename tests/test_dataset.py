from data.dataset import LanguageModelDataset


def test_dataset_shift():

    token_ids = list(range(20))

    dataset = LanguageModelDataset(
        token_ids,
        context_length=5
    )

    input_ids, target_ids = dataset[0]

    assert input_ids.tolist() == [0, 1, 2, 3, 4]
    assert target_ids.tolist() == [1, 2, 3, 4, 5]