from sklearn.utils import shuffle
import pandas

# features = [platform, architecture, packer, option, name, seq1, seq2, seq3 ... seq15]
header = ['platform', 'architecture', 'packer', 'option', 'name', 'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8', 'seq9', 'seq10', 'seq11', 'seq12', 'seq13', 'seq14', 'seq15']
architecture_list = ['32bit', '64bit']
packer_list = ['ASPack', 'ASProtect', 'EnigmaProtector', 'mpress', 'Themida', 'Original', 'Obsidium', 'PESpin', 'UPX', 'VMProtect']


def save_data_to_csv(path, data):
    """
    :param path: string
    :param data: dataFrame
    :return:
    """
    # with open(path, 'w') as f:
    #     pickle.dump(data, f)
    data.columns = header
    data.to_csv(path, index=False, header=True)


# date structure ==> [platform architecture packer option name features*]
def load_csv_data(file_path):
    data = pandas.read_csv(file_path)
    return data
