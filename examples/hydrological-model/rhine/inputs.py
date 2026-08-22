import matplotlib

matplotlib.use('TkAgg')
from pyramids.dataset import DatasetCollection

#%%
root_dir = r"\\MYCLOUDEX2ULTRA\case-studies\rhine\Hapi\Inputs\data\meteodata"
evaporation_data =  rf"{root_dir}\evaporation"

#%%
start_date = "1997_01_01"
end_date = "1997_01_31"
dataset_collection = DatasetCollection.read_multiple_files(
    evaporation_data,
    file_name_data_fmt="%Y_%m_%d",
    regex_string=r"\d{4}_\d{1,2}_\d{1,2}",
    start=start_date,
    end=end_date,
    with_order=True,
    fmt="%Y_%m_%d"
)
#%%
dataset_collection.plot()
