import pydicom
import numpy as np
# 读取DICOM图像文件
dicom_file_path = r"F:\wly\normal control adult\Hu__Wen_Jin_049Y\series0006-Body\img0001--3.60921.dcm"
ds = pydicom.dcmread(dicom_file_path)
# 列出ds对象中的属性
print(dir(ds))
attribute_names = ds.dir()
# 将属性名称与值对应起来并打印
for name in attribute_names:
    if name == "PixelData" or name == "DataSetTrailingPadding":
        continue
        pixel_data_array = np.frombuffer(ds.PixelData, dtype=np.uint8)
        trailing_padding_array = np.frombuffer(ds.DataSetTrailingPadding, dtype=np.uint8)

    value = ds.get(name, "N/A")  # 获取属性的值，如果属性不存在则返回"N/A"
    print(f"{name}: {value}")



# # 打印患者信息
# print("Patient Name:", ds.PatientName)
# print("Patient ID:", ds.PatientID)
# print("Patient Birth Date:", ds.PatientBirthDate)
# print("Patient Sex:", ds.PatientSex)
#
# # 打印图像信息
# print("Instance UID:", ds.SOPInstanceUID)
# print("Series UID:", ds.SeriesInstanceUID)
# print("Image Acquisition Date:", ds.AcquisitionDate)
# print("Image Acquisition Time:", ds.AcquisitionTime)
# print("Manufacturer:", ds.Manufacturer)
# print("Model:", ds.ManufacturerModelName)
#
# # 打印脉冲序列参数
# print("TR:", ds.RepetitionTime)
# print("TE:", ds.EchoTime)
#
# # 打印图像质量信息
# print("Pixel Spacing:", ds.PixelSpacing)
# print("Bits Stored:", ds.BitsStored)
# print("Image Position:", ds.ImagePositionPatient)
#
# # 打印图像备注和描述
# print("Image Comments:", ds.ImageComments)
# print("Image Description:", ds.SeriesDescription)
