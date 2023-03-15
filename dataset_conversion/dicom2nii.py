import SimpleITK as sitk


def dcm2nii(path_read, path_save):
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    # GetGDCMSeriesFileNames读取序列号相同dcm文件的路径，series[0]代表第一个序列号对应的文件
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, path_save)


path_read = r'D:/Datasets/Lung/JiangNan_test'
path_save = r'D:/Datasets/Lung/JiangNan/JiangNan.nii.gz'
dcm2nii(path_read, path_save)



