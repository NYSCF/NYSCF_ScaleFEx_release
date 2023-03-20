import ScaleFEx_class
import numpy as np
#Screens_Emb_Local.Embedding_screens('/media/bianca/Disk3/HTS0012',experiment_name='HTS0012',Plates=np.arange(107,113).astype(str))
#Screens_Vector_Local_NewFeatures12072022.Vector_screens('/media/bianca/Disk21/HTS0026',saving_folder='/home/bianca/Documents/HCI_Projects/Drug_Shifts/Vectors/Refactored/',experiment_name='HTS0026',
                                   # Plates=['102'],parallel=False,downsampling=2)#,SaveImage='/media/bianca/Disk21/HTS0026/HTS0026_Crops/')

ScaleFEx_class.ScaleFEx('/media/bianca/Disk21/HTS0030',saving_folder='/home/bianca/Documents/HCI_Projects/INAD/',experiment_name='HTS0030',
                                    Plates=['101','102','103','104','106'],ROI=150,parallel=False,downsampling=1,visualization=False)

# ScaleFEx_class.Vector_screens('/media/bianca/Disk2/HTS0026',saving_folder='/home/bianca/Documents/HCI_Projects/Drug_Shifts/New/',experiment_name='HTS0026',
#                                     Plates=['101','102','103','104','106'],ROI=150,parallel=False,downsampling=2,visualization=False,use_cpu=False)

# # Screens_Vector_Local.Vector_screens('/media/bianca/Disk2/HTS0026',saving_folder='/home/bianca/Documents/HCI_Projects/Drug_Shifts/New',experiment_name='HTS0026',
# #                                     Plates=['101','102','103','104','106'],ROI=150,parallel=False,downsampling=2,visualization=False,use_cpu=False)
                                    
# ScaleFEx_class.Vector_screens('/media/bianca/Disk2/HTS0026',saving_folder='/home/bianca/Documents/HCI_Projects/Drug_Shifts/New/',experiment_name='HTS0026',
#                                     Plates=['101','102','103','104','106'],ROI=150,parallel=False,downsampling=1,visualization=False,use_cpu=False)


