from CPDPOptuna import CPDPOptuna
from getData import DataProcessing


if __name__ == '__main__':
    data = DataProcessing("Data/KC1.arff", 'Data/Source')
    source_x, source_y, target_x, target_y, loc = data.find_common_metric()

    CPDPOptuna(cpdp_method='Bruakfilter', classifier='KNN', source_x=source_x, source_y=source_y, target_x=target_x,
               target_y=target_y, num_of_times=1).run()
