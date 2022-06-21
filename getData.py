from scipy.io import arff
import os
from typing import List, Tuple
import pandas as pd
import numpy as np


class DataProcessing:
    def __init__(self, target_file: str, source_directory: str):
        self.target_file = target_file
        self.source_directory = source_directory

    @staticmethod
    def load_arff_file(file_name: str) -> Tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame]:
        """
        This function aim to load an arff file and convert it to dataframes.
        :param file_name: a string that have full path with file name to load a specific file
        :return: a tuple of three dataframes that represent data_df, x_data, y_data and a string which is y_name (name of y data column)
        """
        file_data = arff.loadarff(file_name)
        columns: arff.MetaData = file_data[1]
        column_names: list = columns.names()
        column_types: list = columns.types()
        data_df = pd.DataFrame(file_data[0], columns=column_names)
        y_name = column_names[column_types.index('nominal')]
        data_df[y_name] = data_df[y_name].apply(
            lambda x: 1 if x == b'true' or x == b'TRUE' or x == b'Y' or x == b'buggy' else 0)
        x_data = data_df.drop([y_name], axis=1)
        y_data = data_df[y_name]
        y_data.astype(int)
        return data_df, y_name, x_data, y_data

    def get_source_data(self):
        """
        This method aim to load the source projects defect data and add them to self.data list.
        :return: None
        """
        for filename in os.listdir(self.source_directory):
            file_path = os.path.join(self.source_directory, filename)
            if os.path.isfile(file_path):
                data_df, y_name, x_data, y_data = self.load_arff_file(file_path)
                self.data.append((x_data.to_numpy(), y_data.to_numpy(), data_df.columns.tolist()))

        # x_source.replace({np.nan: 0}, inplace=True)
        # y_source.replace({np.nan: 0}, inplace=True)

    def find_common_metric(self, split=False):
        target_df, target_y_name, x_target, y_target = self.load_arff_file(self.target_file)
        self.data: List[Tuple[np.array, np.array, list]] = [(x_target.to_numpy(), y_target.to_numpy(), target_df.columns.tolist())]
        self.get_source_data()

        tx, ty, Ttype = x_target.to_numpy(), y_target.to_numpy(), target_df.columns.tolist()
        tt = tx.shape
        common = []

        # flist = list.copy()
        ### find the common metric
        first = 1
        dump = []
        tmp_data = self.data.copy()
        for item in tmp_data:
            x, y, Stype = item[0], item[1], item[2]
            ss = x.shape

            if first == 1:
                for i in range(ss[1]):
                    if Stype[i] in Ttype:
                        common.append(Stype[i])
                first = 0
            else:
                for i in range(len(common)):
                    if common[i] not in Stype and i not in dump:
                        dump.append(i)
        dump = sorted(dump, reverse=True)
        for i in range(len(dump)):
            common.pop(dump[i])

        ### read the data and concatendate

        if len(common) == 0:
            return 0, 0, 0, 0, []
        else:
            ftx = np.zeros((tt[0], len(common)))
            for i in range(len(common)):
                index = Ttype.index(common[i])
                ftx[:, i] = tx[:, index]

            sx, sy, Stype = tmp_data.pop()

            fsx = np.zeros((len(sy), len(common)))
            for i in range(len(common)):
                index = Stype.index(common[i])
                fsx[:, i] = sx[:, index]

            loc = []
            base = 0

            for item in self.data:
                x, y, Type = item
                loc.append(base)
                base += len(y)
                fx = np.zeros((len(y), len(common)))
                for i in range(len(common)):
                    index = Type.index(common[i])
                    fx[:, i] = x[:, index]
                fsx = np.concatenate((fsx, fx), axis=0)
                sy = np.concatenate((sy, y), axis=0)

            if split:
                return fsx, sy, ftx, ty, loc
            else:
                return fsx, sy, ftx, ty, []