import numpy as np

##### MNIST 2class #######

def load_2c_LR():
    path_lr = "LR/new/2class/"
    central_2c_lr_5e2 = np.load(path_lr + "cent_lr_2c_lr5e-2.npy").T
    central_2c_lr_5e3 = np.load(path_lr + "cent_lr_2c_lr5e-3.npy").T
    central_2c_lr_5e4 = np.load(path_lr + "cent_lr_2c_lr5e-4.npy").T
    central_2c_lr_5e5 = np.load(path_lr + "cent_lr_2c_lr5e-5.npy").T

    global_iid_5e2_file = "iid/MNIST_2classIID_no_comp_LR_lr0.05_lepo1_ba1_global_seed"
    global_iid_5e4_file = "iid/MNIST_2classIID_no_comp_LR_lr0.0005_lepo1_ba1_global_seed"
    global_iid_5e5_file = "iid/MNIST_2classIID_no_comp_LR_lr5e-05_lepo1_ba1_global_seed"

    global_iid_file = "iid/MNIST_2classIID_no_comp_LR_lr0.005_lepo1_ba1_global_seed"
    local_iid_file = "iid/MNIST_2classIID_no_comp_LR_lr0.005_lepo1_ba1_local_seed"
    global_ci_file = "ci/MNIST_2classci_no_comp_LR_lr0.005_lepo1_ba1_global_seed"  
    local_ci_file = "ci/MNIST_2classci_no_comp_LR_lr0.005_lepo1_ba1_local_seed"
    global_sca_file = "ci/MNIST_2classci_scaf_LR_lr0.005_lepo1_ba1_global_seed"  
    local_sca_file = "ci/MNIST_2classci_scaf_LR_lr0.005_lepo1_ba1_local_seed"
    global_si_file = "si/MNIST_2classsi_no_comp_LR_lr0.005_lepo1_ba1_global_seed"
    local_si_file = "si/MNIST_2classsi_no_comp_LR_lr0.005_lepo1_ba1_local_seed"
    global_si_file_wc = "si/MNIST_2classsi_size_comp_LR_lr0.005_lepo1_ba1_global_seed"
    local_si_file_wc = "si/MNIST_2classsi_size_comp_LR_lr0.005_lepo1_ba1_local_seed"


    ci_dgd_file = "ci/MNIST_2classci_dgd_LR_lr0.005_lepo1_ba1_local_seed"
    iid_dgd_file = "iid/MNIST_2classIID_dgd_LR_lr0.005_lepo1_ba1_local_seed"

    lr_2c_iid_g = np.zeros((100,4))
    lr_2c_iid_l = np.zeros((10,100,4))
    lr_2c_iid_5e2 = np.zeros_like(lr_2c_iid_g)
    lr_2c_iid_5e4 = np.zeros_like(lr_2c_iid_g)
    lr_2c_iid_5e5 = np.zeros_like(lr_2c_iid_g)

    lr_2c_ci_g = np.zeros_like(lr_2c_iid_g)
    lr_2c_ci_l = np.zeros_like(lr_2c_iid_l)
    lr_2c_scaf_g = np.zeros_like(lr_2c_iid_g)
    lr_2c_scaf_l = np.zeros_like(lr_2c_iid_l)
    lr_2c_si_g = np.zeros_like(lr_2c_iid_g)
    lr_2c_si_l =  np.zeros_like(lr_2c_iid_l)
    lr_2c_si_wc_g = np.zeros_like(lr_2c_iid_g)
    lr_2c_si_wc_l = np.zeros_like(lr_2c_iid_l)

    lr_2c_ci_dgd = np.zeros_like(lr_2c_iid_l)
    lr_2c_iid_dgd = np.zeros_like(lr_2c_iid_l)

    for i in range(4):
        lr_2c_iid_g[:,i] = np.load(path_lr + global_iid_file + str(i) +  ".npy")[i,:]
        lr_2c_iid_l[:,:,i] = np.load(path_lr + local_iid_file + str(i) +".npy")[i,:,:]
        
        lr_2c_iid_5e2[:,i] = np.load(path_lr + global_iid_5e2_file + str(i) +  ".npy")
        lr_2c_iid_5e4[:,i] = np.load(path_lr + global_iid_5e4_file + str(i) +  ".npy")
        lr_2c_iid_5e5[:,i] = np.load(path_lr + global_iid_5e5_file + str(i) +  ".npy")

        lr_2c_ci_g[:,i]= np.load(path_lr + global_ci_file + str(i) +".npy")
        lr_2c_ci_l[:,:,i] = np.load(path_lr + local_ci_file + str(i) +".npy")
        
        lr_2c_scaf_g[:,i]= np.load(path_lr + global_sca_file + str(i) +".npy")
        lr_2c_scaf_l[:,:,i] = np.load(path_lr + local_sca_file + str(i) +".npy")
        
        lr_2c_si_g[:,i] = np.load(path_lr + global_si_file + str(i) +".npy")[i,:]
        lr_2c_si_l[:,:,i] = np.load(path_lr + local_si_file + str(i) +".npy")[i,:,:]
        lr_2c_si_wc_g[:,i] = np.load(path_lr + global_si_file_wc + str(i) +".npy")[i,:]
        lr_2c_si_wc_l[:,:,i]= np.load(path_lr + local_si_file_wc + str(i) +".npy")[i,:,:]
        
        lr_2c_ci_dgd[:,:,i] = np.load(path_lr + ci_dgd_file + str(i) + ".npy")
        lr_2c_iid_dgd[:,:,i] = np.load(path_lr + iid_dgd_file + str(i) + ".npy")

    to_plot_lr = {
        "iid 5e2" :lr_2c_iid_5e2,
        "iid 5e3" :lr_2c_iid_g,
        "iid 5e4" :lr_2c_iid_5e4,
        "iid 5e5" :lr_2c_iid_5e5,
        "ci"      :lr_2c_ci_g,
        "scaf"    :lr_2c_scaf_g,
        "si"      :lr_2c_si_wc_g,
        "ci dgd"  :lr_2c_ci_dgd,
        "iid dgd" :lr_2c_iid_dgd,
        "central 5e2": central_2c_lr_5e2,
        "central 5e3": central_2c_lr_5e3,
        "central 5e4": central_2c_lr_5e4,
        "central 5e5": central_2c_lr_5e5 
    }
    #to_plot_lr  = [lr_2c_iid_g, lr_2c_ci_g, lr_2c_si_wc_g, central_2c_lr_5e3, central_2c_lr_5e4]
    return(to_plot_lr)

def load_2c_SVM():
    path_svm = "SVM/new/2class/"

    central_2c_svm_5e4 = np.load(path_svm+ "cent_svm_2c_lr5e-4npy").T
    central_2c_svm_5e5 = np.load(path_svm+ "cent_svm_2c_lr5e-5.npy").T
    #central_2c_5e6 = np.load(path_svm+ "cent_svm_2c_lr5e-6.npy")


    global_iid_file = "iid/MNIST_2classIID_no_comp_SVM_lr0.0005_lepo1_ba1_global_seed"
    local_iid_file = "iid/MNIST_2classIID_no_comp_SVM_lr0.0005_lepo1_ba1_local_seed"
    global_ci_file = "ci/MNIST_2classci_no_comp_SVM_lr0.0005_lepo1_ba1_global_seed"  
    local_ci_file = "ci/MNIST_2classci_no_comp_SVM_lr0.0005_lepo1_ba1_local_seed"
    global_sca_file = "ci/MNIST_2classci_scaf_SVM_lr0.0005_lepo1_ba1_global_seed"
    global_si_file = "si/MNIST_2classsi_no_comp_SVM_lr0.0005_lepo1_ba1_global_seed"
    local_si_file = "si/MNIST_2classsi_no_comp_SVM_lr0.0005_lepo1_ba1_local_seed"
    global_si_file_wc = "si/MNIST_2classsi_size_comp_SVM_lr0.0005_lepo1_ba1_global_seed"
    local_si_file_wc = "si/MNIST_2classsi_size_comp_SVM_lr0.0005_lepo1_ba1_local_seed"

    iid_dgd_file = "iid/MNIST_2classIID_dgd_SVM_lr0.0005_lepo1_ba1_local_seed"

    svm_2c_iid_g = np.zeros((100,4))
    svm_2c_iid_l = np.zeros((10,100,4))
    svm_2c_ci_g = np.zeros_like(svm_2c_iid_g)
    svm_2c_ci_l = np.zeros_like(svm_2c_iid_l)
    svm_2c_sca_g = np.zeros_like(svm_2c_iid_g)
    svm_2c_si_g = np.zeros_like(svm_2c_iid_g)
    svm_2c_si_l =  np.zeros_like(svm_2c_iid_l)
    svm_2c_si_wc_g = np.zeros_like(svm_2c_iid_g)
    svm_2c_si_wc_l = np.zeros_like(svm_2c_iid_l)

    svm_2c_iid_dgd = np.zeros_like(svm_2c_iid_l)

    for i in range(4):
        svm_2c_iid_g[:,i] = np.load(path_svm+ global_iid_file + str(i) +  ".npy")[i,:]

        svm_2c_sca_g[:,i] = np.load(path_svm+ global_sca_file + str(i) +".npy")
        
        svm_2c_si_g[:,i] = np.load(path_svm+ global_si_file + str(i) +".npy")[i,:]
        svm_2c_si_l[:,:,i] = np.load(path_svm+ local_si_file + str(i) +".npy")[i,:,:]
        svm_2c_si_wc_g[:,i] = np.load(path_svm+ global_si_file_wc + str(i) +".npy")[i,:]
        svm_2c_si_wc_l[:,:,i]= np.load(path_svm+ local_si_file_wc + str(i) +".npy")[i,:,:]
        
        svm_2c_iid_dgd[:,:,i]= np.load(path_svm + iid_dgd_file + str(i) + ".npy")

    for i in range(1,4):
        svm_2c_ci_g[:,i]= np.load(path_svm+ global_ci_file + str(i) +".npy")[i-1,:]
        svm_2c_ci_l[:,:,i] = np.load(path_svm+ local_ci_file + str(i) +".npy")[i-1, :]
        
    svm_2c_ci_g[:,0]= np.load(path_svm+ global_ci_file + str(0) +".npy")[0,:]
    svm_2c_ci_l[:,:,0] = np.load(path_svm+ local_ci_file + str(0) +".npy")[0, :]

    to_plot_svm = {
        "iid" : svm_2c_iid_g,
        "ci"  : svm_2c_ci_g,
        "scaf": svm_2c_sca_g,
        "si"  : svm_2c_si_wc_g,
        "iid dgd": svm_2c_iid_dgd,
        "cent 5e4": central_2c_svm_5e4,
        "cent 5e5": central_2c_svm_5e5
    }
    #to_plot_svm = [svm_2c_iid_g, svm_2c_ci_g, svm_2c_si_g,  central_2c_svm_5e4, central_2c_svm_5e5 ]

    return to_plot_svm

def load_2c_DT():
    path = "GBDT/2class/"
    central_dt = np.load(path + "trees_2class_centralized.npy").T

    global_iid_file = "iid/Trees_IID_2Classglobal_seed"
    local_iid_file ="iid/Trees_IID_2Classlocal_seed"
    global_ci_file = "ci/Trees_ci_2Classglobal_seed"
    local_ci_file = "ci/Trees_ci_2Classlocal_seed"
    global_si_file = "si/Trees_si_2Classglobal_seed"
    local_si_file = "si/Trees_si_2Classlocal_seed"


    dt_2c_iid_g = np.zeros((100,4))
    dt_2c_iid_l = np.zeros((100,4))
    dt_2c_ci_g = np.zeros_like(dt_2c_iid_g)
    dt_2c_ci_l = np.zeros_like(dt_2c_iid_l)
    dt_2c_si_g = np.zeros_like(dt_2c_iid_g)
    dt_2c_si_l =  np.zeros_like(dt_2c_iid_l)

    for i in range(4):
        dt_2c_iid_g[:,i] = np.load(path + global_iid_file + str(i) +  ".npy")[i,:]
        dt_2c_iid_l[:,i] = np.load(path + local_iid_file + str(i) +".npy")[i,:]
        dt_2c_ci_g[:,i]= np.load(path + global_ci_file + str(i) +".npy")[i,:]
        dt_2c_ci_l[:,i] = np.load(path + local_ci_file + str(i) +".npy")[i,:]
        dt_2c_si_g[:,i] = np.load(path + global_si_file + str(i) +".npy")[i,:]
        dt_2c_si_l[:,i] = np.load(path + local_si_file + str(i) +".npy")[i,:]

    to_plot_dt = {
        "iid" : dt_2c_iid_g,
        "ci"  : dt_2c_ci_g,
        "si"  : dt_2c_si_g,
        "cent": central_dt
    } 
    return to_plot_dt

def load_2c_FNN():
    path = "FNN/2class/"

    central_2c_5e1 = np.load(path + "central_FNN_b1.npy")
    central_2c_5e2 = np.load(path + "central_FNN_lr5e-2.npy")

    FNN_2c_iid_g = np.load(path + "IID/MNIST_2ClassIID_nc_FNN_global_seed37.npy" )
    FNN_2c_iid_l = np.load(path + "IID/MNIST_2ClassIID_nc_FNN_local_seed37.npy" )

    FNN_2c_ci_nc_g = np.load(path + "ci/MNIST_2Classci_nc_FNN_global_seed100_103.npy")
    FNN_2c_ci_nc_l = np.load(path + "ci/MNIST_2Class_ci_nc_FNN_local_seed100_103.npy")
    ci_sca_g_file = "ci/MNIST_2classci_scaf_FNN_lr0.5_lepo1_ba1_global_seed"
    ci_sca_l_file = "ci/MNIST_2classci_scaf_FNN_lr0.5_lepo1_ba1local_seed"
    si_nc_g_file = "si/MNIST_2classsi_no_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    si_nc_l_file = "si/MNIST_2classsi_no_comp_FNN_lr0.5_lepo1_ba1local_seed"
    si_wc_g_file = "si/MNIST_2classsi_size_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    si_wc_l_file = "si/MNIST_2classsi_size_comp_FNN_lr0.5_lepo1_ba1local_seed"
    dgd_iid_file = "IID/MNIST_2classIID_dgd_FNN_lr0.5_lepo1_ba1local_seed"

    FNN_2c_ci_sca_g = np.zeros((4,100))
    FNN_2c_ci_sca_l = np.zeros((4,10,100))
    FNN_2c_si_nc_g = np.zeros_like(FNN_2c_ci_sca_g)
    FNN_2c_si_nc_l = np.zeros_like(FNN_2c_ci_sca_l)
    FNN_2c_si_wc_g = np.zeros_like(FNN_2c_ci_sca_g)
    FNN_2c_si_wc_l = np.zeros_like(FNN_2c_ci_sca_l)
    FNN_2c_iid_dgd_g = np.zeros((4,10,100))


    for i in range(4):
        FNN_2c_ci_sca_g[i,:] = np.load(path + ci_sca_g_file + str(i) + ".npy")
        FNN_2c_ci_sca_l[i,:,:] = np.load(path + ci_sca_l_file + str(i) + ".npy")
        FNN_2c_si_nc_g[i,:] = np.load(path + si_nc_g_file + str(i) + ".npy")
        FNN_2c_si_nc_l[i,:,:] = np.load(path + si_nc_l_file + str(i) + ".npy")
        FNN_2c_si_wc_g[i,:] = np.load(path + si_wc_g_file + str(i) + ".npy")
        FNN_2c_si_wc_l[i,:,:] = np.load(path + si_wc_l_file + str(i) + ".npy")
        FNN_2c_iid_dgd_g[i,:,:] = np.load(path + dgd_iid_file + str(i) + ".npy")

    to_plot_FNN = {
        "iid" : FNN_2c_iid_g,
        "iid loc" : FNN_2c_iid_l,
        "ci"  : FNN_2c_ci_nc_g,
        "scaf": FNN_2c_ci_sca_g,
        "dgd" : FNN_2c_iid_dgd_g,
        "si"  : FNN_2c_si_wc_g,
        "cent 5e1": central_2c_5e1,
        "cent 5e2": central_2c_5e2
    }
    #to_plot_FNN = [FNN_2c_iid_g, FNN_2c_ci_nc_g, FNN_2c_si_wc_g, central_2c_5e1, central_2c_5e2]
    return to_plot_FNN

def load_2c_CNN():
    path = "CNN/2class/"

    central_cnn_2c_5e2 = np.load(path + "central_cnn_2class_10ba.npy")
    central_cnn_2c_5e3 = np.load(path + "central_cnn_2c_lr5e-3.npy")

    cnn_2c_iid_g_file = "IID/IID_no_comp_CNN_lr0.05_lepo1_ba10CNN_global_seed"
    cnn_2c_iid_l_file = "IID/IID_no_comp_CNN_lr0.05_lepo1_ba10CNN_local_seed"

    ci_nc_g_file = "ci/ci_no_comp_CNN_lr0.05_lepo1_ba10CNN_global_seed"
    ci_nc_l_file = "ci/ci_no_comp_CNN_lr0.05_lepo1_ba10CNN_local_seed"
    ci_sca_g_file = "ci/MNIST_2classci_scaf_CNN_lr0.05_lepo1_ba10_global_seed" 
    ci_sca_l_file = "ci/MNIST_2classci_scaf_CNN_lr0.05_lepo1_ba10local_seed" 
    si_nc_g_file = "si/si_no_compglobal_seed"
    si_nc_l_file = "si/si_no_complocal_seed"
    si_wc_g_file = "si/MNIST_2classsi_size_comp_CNN_lr0.05_lepo1_ba10_global_seed"
    si_wc_l_file = "si/MNIST_2classsi_size_comp_CNN_lr0.05_lepo1_ba10local_seed"

    cnn_2c_iid_g = np.zeros((4,100))
    cnn_2c_iid_l = np.zeros_like(cnn_2c_iid_g)
    cnn_2c_ci_nc_g = np.zeros_like(cnn_2c_iid_g)
    cnn_2c_ci_nc_l = np.zeros_like(cnn_2c_iid_g)
    cnn_2c_ci_sca_g = np.zeros((4,100))
    cnn_2c_ci_sca_l = np.zeros_like(cnn_2c_ci_sca_g)
    cnn_2c_si_nc_g = np.zeros_like(cnn_2c_ci_sca_g)
    cnn_2c_si_nc_l = np.zeros_like(cnn_2c_ci_sca_g)
    cnn_2c_si_wc_g = np.zeros_like(cnn_2c_ci_sca_g)
    cnn_2c_si_wc_l = np.zeros_like(cnn_2c_ci_sca_g)


    for i in range(4):
        cnn_2c_iid_g[i,:] = np.load(path + cnn_2c_iid_g_file + str(i) + ".npy")
        cnn_2c_iid_l[i,:] = np.mean(np.load(path + cnn_2c_iid_l_file + str(i) + ".npy"), axis = 0)
        cnn_2c_ci_nc_g[i,:] = np.load(path+ ci_nc_g_file + str(i) + ".npy")
        cnn_2c_ci_nc_l[i,:] = np.mean(np.load(path+ ci_nc_l_file + str(i) + ".npy"), axis=0)
        cnn_2c_ci_sca_g[i,:] = np.load(path + ci_sca_g_file + str(i) + ".npy")
        cnn_2c_ci_sca_l[i,:] = np.mean(np.load(path + ci_sca_l_file + str(i) + ".npy"), axis=0)
        cnn_2c_si_nc_g[i,:] = np.load(path + si_nc_g_file + str(i) + ".npy")[i,:]
        cnn_2c_si_nc_l[i,:] = np.mean(np.load(path + si_nc_l_file + str(i) + ".npy"), axis=1)[i,:]
        cnn_2c_si_wc_g[i,:] = np.load(path + si_wc_g_file + str(i) + ".npy")
        cnn_2c_si_wc_l[i,:] = np.mean(np.load(path + si_wc_l_file + str(i) + ".npy"), axis=0)

    to_plot_cnn_2c = {
        "iid" : cnn_2c_iid_g,
        "ci"  : cnn_2c_ci_nc_g,
        "scaf" : cnn_2c_ci_sca_g,
        "si"  : cnn_2c_si_wc_g,
        "cent 5e2": central_cnn_2c_5e2,
        "cent 5e3": central_cnn_2c_5e3
    }    
    #to_plot_cnn_2c = [cnn_2c_iid_g, cnn_2c_ci_nc_g, cnn_2c_si_wc_g, central_cnn_2c_5e2, central_cnn_2c_5e3]    

    return to_plot_cnn_2c

###### MNIST 4class ######

def load_4c_LR():
    path = "LR/new/4class/"

    central_4c_5e4 = np.load(path + "cent_4c_lr_lr5e-4.npy").T
    central_4c_5e5 = np.load(path + "cent_4c_lr_lr5e-5.npy").T

    global_iid_file = "iid/MNIST_4classIID_no_comp_LR_lr0.0005_lepo1_ba1_global_seed"
    local_iid_file = "iid/MNIST_4classIID_no_comp_LR_lr0.0005_lepo1_ba1_local_seed"
    global_ci_file = "ci/MNIST_4classci_size_comp_LR_lr0.0005_lepo1_ba1_global_seed"  
    local_ci_file = "ci/MNIST_4classci_size_comp_LR_lr0.0005_lepo1_ba1_local_seed"

    global_sca_file = "ci/MNIST_4classci_scaf_LR_lr0.0005_lepo1_ba1_global_seed"  
    local_sca_file = "ci/MNIST_4classci_scaf_LR_lr0.0005_lepo1_ba1_local_seed"

    global_si_file = "si/MNIST_4classsi_no_comp_LR_lr0.0005_lepo1_ba1_global_seed"
    local_si_file = "si/MNIST_4classsi_no_comp_LR_lr0.0005_lepo1_ba1_local_seed"
    global_si_file_wc = "si/MNIST_4classsi_size_comp_LR_lr0.0005_lepo1_ba1_global_seed"
    local_si_file_wc = "si/MNIST_4classsi_size_comp_LR_lr0.0005_lepo1_ba1_local_seed"

    lr_4c_iid_g = np.zeros((100,4))
    lr_4c_iid_l = np.zeros((10,100,4))
    lr_4c_ci_g = np.zeros_like(lr_4c_iid_g)
    lr_4c_ci_l = np.zeros_like(lr_4c_iid_l)
    lr_4c_scaf_g = np.zeros_like(lr_4c_iid_g)
    lr_4c_scaf_l = np.zeros_like(lr_4c_iid_l)
    lr_4c_si_g = np.zeros_like(lr_4c_iid_g)
    lr_4c_si_l =  np.zeros_like(lr_4c_iid_l)
    lr_4c_si_wc_g = np.zeros_like(lr_4c_iid_g)
    lr_4c_si_wc_l = np.zeros_like(lr_4c_iid_l)


    for i in range(4):
        lr_4c_iid_g[:,i] = np.load(path + global_iid_file + str(i) +  ".npy")
        lr_4c_iid_l[:,:,i] = np.load(path + local_iid_file + str(i) +".npy")
        
        lr_4c_ci_g[:,i]= np.load(path + global_ci_file + str(i) +".npy")
        lr_4c_ci_l[:,:,i] = np.load(path + local_ci_file + str(i) +".npy")
        
        lr_4c_scaf_g[:,i]= np.load(path + global_sca_file + str(i) +".npy")
        lr_4c_scaf_l[:,:,i] = np.load(path + local_sca_file + str(i) +".npy")

    for i in range(2):
        lr_4c_si_wc_g[:,i] = np.load(path + global_si_file_wc + str(i) +".npy")
        lr_4c_si_wc_l[:,:,i] = np.load(path + local_si_file_wc + str(i) +".npy")

    for i in range(4,6):
        lr_4c_si_wc_g[:,i-2] = np.load(path + global_si_file_wc + str(i) +".npy")
        lr_4c_si_wc_l[:,:,i-2] = np.load(path + local_si_file_wc + str(i) +".npy")     
        
    for i in range(3,7):
        lr_4c_si_g[:,i-3] = np.load(path + global_si_file + str(i) +".npy")
        lr_4c_si_l[:,:,i-3] = np.load(path + local_si_file + str(i) +".npy")
        
    #to_plot_lr = [lr_4c_iid_g, lr_4c_ci_g, lr_4c_si_wc_g, central_4c_5e4, central_4c_5e5]
    to_plot_lr = {

    "iid"       :lr_4c_iid_g,
    "ci"        :lr_4c_ci_g,
    "scaf"      :lr_4c_scaf_g,
    "si"        :lr_4c_si_wc_g,
    "central 5e4": central_4c_5e4,
    "central 5e5": central_4c_5e5 
    }
    return to_plot_lr

def load_4c_SVM():
    path = "SVM/new/4class/"

    central_4c_5e5 = np.load(path + "cent_svm_4c.npy").T
    central_4c_5e6 = np.load(path + "cent_svm_4c_5e-6.npy").T

    global_iid_file = "iid/MNIST_4classIID_no_comp_SVM_lr5e-05_lepo1_ba1_global_seed"
    local_iid_file = "iid/MNIST_4classIID_no_comp_SVM_lr5e-05_lepo1_ba1_local_seed"
    global_ci_file = "ci/MNIST_4classci_no_comp_SVM_lr5e-05_lepo1_ba1_global_seed"  
    local_ci_file = "ci/MNIST_4classci_no_comp_SVM_lr5e-05_lepo1_ba1_local_seed"
    global_sca_file = "ci/MNIST_4classci_scaf_SVM_lr5e-05_lepo1_ba1_global_seed"
    local_sca_file = "ci/MNIST_4classci_scaf_SVM_lr5e-05_lepo1_ba1_local_seed"
    global_si_file = "si/MNIST_4classsi_no_comp_SVM_lr5e-05_lepo1_ba1_global_seed"
    local_si_file = "si/MNIST_4classsi_no_comp_SVM_lr5e-05_lepo1_ba1_local_seed"
    global_si_file_wc = "si/MNIST_4classsi_size_comp_SVM_lr5e-05_lepo1_ba1_global_seed"
    local_si_file_wc = "si/MNIST_4classsi_size_comp_SVM_lr5e-05_lepo1_ba1_local_seed"

    svm_4c_iid_g = np.zeros((100,4))
    svm_4c_iid_l = np.zeros((10,100,4))
    svm_4c_ci_g = np.zeros_like(svm_4c_iid_g)
    svm_4c_ci_l = np.zeros_like(svm_4c_iid_l)
    svm_4c_sca_g = np.zeros_like(svm_4c_iid_g)
    svm_4c_sca_l =  np.zeros_like(svm_4c_iid_l)
    svm_4c_si_g = np.zeros_like(svm_4c_iid_g)
    svm_4c_si_l =  np.zeros_like(svm_4c_iid_l)
    svm_4c_si_wc_g = np.zeros_like(svm_4c_iid_g)
    svm_4c_si_wc_l = np.zeros_like(svm_4c_iid_l)

    for i in range(4):
        svm_4c_iid_g[:,i] = np.load(path + global_iid_file + str(i) +  ".npy")
        svm_4c_iid_l[:,:,i] = np.load(path + local_iid_file + str(i) +".npy")

        svm_4c_sca_g[:,i] = np.load(path + global_sca_file + str(i) +".npy")
        svm_4c_sca_l[:,:,i] = np.load(path + local_sca_file + str(i) +".npy")
        
        svm_4c_ci_g[:,i] = np.load(path + global_ci_file + str(i) +".npy")
        svm_4c_ci_l[:,:,i] = np.load(path + local_ci_file + str(i) +".npy")
            
        svm_4c_si_g[:,i] = np.load(path + global_si_file + str(i) +".npy")
        svm_4c_si_l[:,:,i] = np.load(path + local_si_file + str(i) +".npy")
        svm_4c_si_wc_g[:,i] = np.load(path + global_si_file_wc + str(i) +".npy")
        svm_4c_si_wc_l[:,:,i]= np.load(path + local_si_file_wc + str(i) +".npy")
        
    to_plot_svm = [svm_4c_iid_g, svm_4c_ci_g,  svm_4c_si_wc_g, central_4c_5e5, central_4c_5e6 ]
    return to_plot_svm

def load_4c_DT():
    path = "GBDT/4class/"
    central_DT = np.load(path + "trees_4class_centralized.npy")

    global_iid_file = "iid/Trees_IID_4Classglobal_seed"
    local_iid_file ="iid/Trees_IID_4Classlocal_seed"
    global_ci_file = "ci/Trees_ci_4Classglobal_seed"
    local_ci_file = "ci/Trees_ci_4Classlocal_seed"
    global_si_file = "si/Trees_si_4Classglobal_seed"
    local_si_file = "si/Trees_si_4Classlocal_seed"


    dt_4c_iid_g = np.zeros((100,4))
    dt_4c_iid_l = np.zeros((100,4))
    dt_4c_ci_g = np.zeros_like(dt_4c_iid_g)
    dt_4c_ci_l = np.zeros_like(dt_4c_iid_l)
    dt_4c_si_g = np.zeros_like(dt_4c_iid_g)
    dt_4c_si_l =  np.zeros_like(dt_4c_iid_l)

    for i in range(4):
        dt_4c_iid_g[:,i] = np.load(path + global_iid_file + str(i) +  ".npy")[i,:]
        dt_4c_iid_l[:,i] = np.load(path + local_iid_file + str(i) +".npy")[i,:]
        dt_4c_ci_g[:,i]= np.load(path + global_ci_file + str(i) +".npy")[i,:]
        dt_4c_ci_l[:,i] = np.load(path + local_ci_file + str(i) +".npy")[i,:]
        dt_4c_si_g[:,i] = np.load(path + global_si_file + str(i) +".npy")[i,:]
        dt_4c_si_l[:,i] = np.load(path + local_si_file + str(i) +".npy")[i,:]
        
    to_plot_DT = [dt_4c_iid_g, dt_4c_ci_g, dt_4c_si_g, central_DT]
    return to_plot_DT

def load_4c_FNN():
    path = "FNN/4class/"

    central_FNN_5e1 = np.load(path + "4c_fnn_central_noba.npy")
    central_FNN_5e2 = np.load(path + "4c_fnn_central_5e-2.npy")

    iid_g_file = "IID/MNIST_4classIID_no_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    iid_l_file = "IID/MNIST_4classIID_no_comp_FNN_lr0.5_lepo1_ba1local_seed"

    ci_nc_g_file = "ci/MNIST_4classci_no_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    ci_nc_l_file = "ci/MNIST_4classci_no_comp_FNN_lr0.5_lepo1_ba1local_seed"

    ci_sca_g_file = "ci/MNIST_4classci_scaf_FNN_lr0.5_lepo1_ba1_global_seed"
    ci_sca_l_file = "ci/MNIST_4classci_scaf_FNN_lr0.5_lepo1_ba1local_seed"

    si_nc_g_file = "si/MNIST_4classsi_no_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    si_nc_l_file = "si/MNIST_4classsi_no_comp_FNN_lr0.5_lepo1_ba1local_seed"
    si_wc_g_file = "si/MNIST_4classsi_size_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    si_wc_l_file = "si/MNIST_4classsi_size_comp_FNN_lr0.5_lepo1_ba1local_seed"



    FNN_ci_sca_g = np.zeros((4,100))
    FNN_ci_sca_l = np.zeros_like(FNN_ci_sca_g)
    FNN_ci_nc_g = np.zeros_like(FNN_ci_sca_g)
    FNN_ci_nc_l = np.zeros_like(FNN_ci_sca_g)
    FNN_si_nc_g = np.zeros_like(FNN_ci_sca_g)
    FNN_si_nc_l = np.zeros_like(FNN_ci_sca_g)
    FNN_si_wc_g = np.zeros_like(FNN_ci_sca_g)
    FNN_si_wc_l = np.zeros_like(FNN_ci_sca_g)
    FNN_iid_g = np.zeros_like(FNN_ci_sca_g)
    FNN_iid_l = np.zeros_like(FNN_ci_sca_g)

    for i in range(4):
        FNN_iid_g[i,:] = np.load(path + iid_g_file + str(i) + ".npy")
        FNN_iid_l[i,:] = np.mean(np.load(path + iid_l_file + str(i) + ".npy"), axis=0)
        FNN_ci_sca_g[i,:] = np.load(path + ci_sca_g_file + str(i) + ".npy")
        FNN_ci_sca_l[i,:] = np.mean(np.load(path + ci_sca_l_file + str(i) + ".npy"), axis=0)
        FNN_ci_nc_g[i,:] = np.load(path + ci_nc_g_file + str(i) + ".npy")
        FNN_ci_nc_l[i,:] = np.mean(np.load(path + ci_nc_l_file + str(i) + ".npy"), axis=0)
        FNN_si_nc_g[i,:] = np.load(path + si_nc_g_file + str(i) + ".npy")
        FNN_si_nc_l[i,:] = np.mean(np.load(path + si_nc_l_file + str(i) + ".npy"), axis=0)
        FNN_si_wc_g[i,:] = np.load(path + si_wc_g_file + str(i) + ".npy")
        FNN_si_wc_l[i,:] = np.mean(np.load(path + si_wc_l_file + str(i) + ".npy"), axis=0)
        
    to_plot_FNN = [FNN_iid_g, FNN_ci_nc_g, FNN_ci_sca_g, FNN_si_nc_g, FNN_si_wc_g, central_FNN_5e1, central_FNN_5e2]
    #to_plot_FNN = [FNN_iid_g, FNN_ci_nc_g, FNN_si_wc_g, central_FNN_5e1, central_FNN_5e2]
    to_plot_FNN = {
        "iid" : FNN_iid_g,
        "ci"  : FNN_ci_nc_g,
        "scaf": FNN_ci_sca_g,
        "si"  : FNN_si_wc_g,
        "cent 5e1": central_FNN_5e1,
        "cent 5e2": central_FNN_5e2
    }
    return to_plot_FNN

def load_4c_CNN():
    path = "CNN/4class/"

    central_CNN_5e2 = np.load(path + "4c_cnn_central_5e-2.npy")
    central_CNN_5e3 = np.load(path + "4c_cnn_central_5e-3.npy")

    iid_g_file = "IID/MNIST_4classIID_no_comp_CNN_lr0.05_lepo1_ba10CNN_global_seed"
    iid_l_file = "IID/MNIST_4classIID_no_comp_CNN_lr0.05_lepo1_ba10CNN_local_seed"

    ci_nc_g_file = "ci/MNIST_4classci_no_comp_CNN_lr0.05_lepo1_ba10_global_seed"
    ci_nc_l_file = "ci/MNIST_4classci_no_comp_CNN_lr0.05_lepo1_ba10local_seed"
    ci_sca_g_file = "ci/MNIST_4classci_scaf_CNN_lr0.05_lepo1_ba10_global_seed" 
    ci_sca_l_file = "ci/MNIST_4classci_scaf_CNN_lr0.05_lepo1_ba10local_seed" 
    si_nc_g_file = "si/MNIST_4classsi_no_comp_CNN_lr0.05_lepo1_ba10_global_seed"
    si_nc_l_file = "si/MNIST_4classsi_no_comp_CNN_lr0.05_lepo1_ba10local_seed"
    si_wc_g_file = "si/MNIST_4classsi_size_comp_CNN_lr0.05_lepo1_ba10_global_seed"
    si_wc_l_file = "si/MNIST_4classsi_size_comp_CNN_lr0.05_lepo1_ba10local_seed"


    CNN_iid_g = np.zeros((4,100))
    CNN_iid_l = np.zeros_like(CNN_iid_g)
    CNN_ci_nc_g = np.zeros((4,100))
    CNN_ci_nc_l = np.zeros((4,100))
    CNN_ci_sca_g = np.zeros((4,100))
    CNN_ci_sca_l = np.zeros_like(CNN_ci_sca_g)
    CNN_si_nc_g = np.zeros((4,200))
    CNN_si_nc_l = np.zeros_like(CNN_si_nc_g)
    CNN_si_wc_g = np.zeros_like(CNN_si_nc_g)
    CNN_si_wc_l = np.zeros_like(CNN_si_nc_g)


    for i in range(4):
        CNN_iid_g[i,:] = np.load(path + iid_g_file + str(i) + ".npy")
        CNN_iid_l[i,:] = np.mean(np.load(path + iid_l_file + str(i) + ".npy"), axis = 0)

        CNN_ci_nc_g[i,:] = np.load(path+ ci_nc_g_file + str(i) + ".npy")
        CNN_ci_nc_l[i,:] = np.mean(np.load(path+ ci_nc_l_file + str(i) + ".npy"), axis=0)
    
        CNN_ci_sca_g[i,:] = np.load(path + ci_sca_g_file + str(i) + ".npy")
        CNN_ci_sca_l[i,:] = np.mean(np.load(path + ci_sca_l_file + str(i) + ".npy"), axis=0)
        CNN_si_nc_g[i,:] = np.load(path + si_nc_g_file + str(i) + ".npy")
        CNN_si_nc_l[i,:] = np.mean(np.load(path + si_nc_l_file + str(i) + ".npy"), axis=0)
        CNN_si_wc_g[i,:] = np.load(path + si_wc_g_file + str(i) + ".npy")
        CNN_si_wc_l[i,:] = np.mean(np.load(path + si_wc_l_file + str(i) + ".npy"), axis=0)

    to_plot_CNN = [CNN_iid_g, CNN_ci_nc_g, CNN_si_wc_g, central_CNN_5e2, central_CNN_5e3]
    return to_plot_CNN

###### Fashion MNIST #####


def load_f_LR():

    path1 = "LR/new/fashion_MNIST/"
    path2 = "LR/new/fashion_MNIST_ci/"

    central_f_5e5 = np.load(path1 + "cent_fashion_MNIST_lr_lr5e-5.npy").T
    central_f_5e6 = np.load(path1 + "cent_fashion_MNIST_lr_lr5e-6.npy").T

    global_iid_file = "fashion_MNISTIID_no_comp_LR_lr5e-05_lepo1_ba1_global_seed"
    local_iid_file = "fashion_MNISTIID_no_comp_LR_lr5e-05_lepo1_ba1_local_seed"
    f_ci_g_file = "fashion_MNISTci_size_comp_LR_lr5e-05_lepo1_ba1_global_seed"
    f_ci_l_file = "fashion_MNISTci_size_comp_LR_lr5e-05_lepo1_ba1_local_seed"


    lr_f_iid_g = np.zeros((200,4))
    lr_f_iid_l = np.zeros((10,200,4))
    lr_f_ci_g = np.zeros((100,4))
    lr_f_ci_l = np.zeros((10,100,4))


    for i in range(4):
        lr_f_ci_g[:,i] = np.load(path2 + f_ci_g_file + str(i) + ".npy")
        lr_f_ci_l[:,:,i] = np.load(path2 + f_ci_l_file + str(i) + ".npy")
        lr_f_iid_g[:,i] = np.load(path1 + global_iid_file + str(i) +  ".npy")
        lr_f_iid_l[:,:,i] = np.load(path1 + local_iid_file + str(i) +".npy")

    #to_plot_lr = [lr_f_iid_g, lr_f_ci_g.T, central_f_5e5, central_f_5e6 ]
    to_plot_lr = {

    "iid"       :lr_f_iid_g,
    "ci"        :lr_f_ci_g,
    "central 5e5": central_f_5e5,
    "central 5e6": central_f_5e6 
    }
    return to_plot_lr

def load_f_SVM():
    pass

def load_f_DT():
    path = "GBDT/fashion MNIST/"
    path2 = "GBDT/fashion MNIST ci/"
    central_f = np.load(path + "trees_fashion_MNIST_centralized.npy").T

    global_iid_file = "Trees_fashion_MNISTglobal_seed"
    local_iid_file ="Trees_fashion_MNISTlocal_seed"

    global_ci_file = "Trees_fashion_MNIST_CIglobal_seed" 
    local_ci_file = "Trees_fashion_MNIST_CIlocal_seed"

    global_ci_updated_file = "Trees_fashion_MNIST_CI_updatedglobal_seed"
    local_ci_updated_file = "Trees_fashion_MNIST_CI_updatedlocal_seed"


    dt_iid_g = np.zeros((100,4))
    dt_iid_l = np.zeros((100,4))

    dt_ci_g = np.zeros((100,4))
    dt_ci_l = np.zeros((100,4))
        
    dt_ci_u_g = np.zeros((100,4))
    dt_ci_u_l = np.zeros((100,4))


    for i in range(4):
        dt_iid_g[:,i] = np.load(path + global_iid_file + str(i) +  ".npy")[i,:]
        dt_iid_l[:,i] = np.load(path + local_iid_file + str(i) +".npy")[i,:]
        
        dt_ci_g[:,i] = np.load(path2 + global_ci_file + str(i) + ".npy")[i,:]
        dt_ci_l[:,i] = np.load(path2 + local_ci_file + str(i) + ".npy")[i,:]

        dt_ci_u_g[:,i] = np.load(path2 + global_ci_updated_file + str(i) + ".npy")[i,:]
        dt_ci_u_l[:,i] = np.load(path2 + local_ci_updated_file + str(i) + ".npy")[i,:]

        
    to_plot_dt = {
        "iid" : dt_iid_g,
        "ci"  : dt_ci_g,
        "ci updated" : dt_ci_u_g,
        "ci local updated" : dt_ci_u_l,
        "cent": central_f
    } 
    return to_plot_dt

def load_f_FNN():
    path = "FNN/fashion_MNIST/"
    #new_path = "/home/swier/Documents/afstuderen/datafiles/nn/fashion/"
    


    cent_f = np.load(path + "central_fnn_fashion_MNIST_lr5e-1.npy")

    f_ci_file_g = "fashion_MNISTci_size_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    f_ci_file_l = "fashion_MNISTci_size_comp_FNN_lr0.5_lepo1_ba1local_seed"
    f_file_g = "fashion_MNISTIID_no_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    f_file_l = "fashion_MNISTIID_no_comp_FNN_lr0.5_lepo1_ba1local_seed"
    f_ci_file_g = "../fashion_MNIST_ci/fashion_MNISTci_size_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    f_ci_file_l = "../fashion_MNIST_ci/fashion_MNISTci_size_comp_FNN_lr0.5_lepo1_ba1local_seed"


    #new_iid_file_l = "fashion_MNISTIID_size_comp_FNN_lr0.5_lepo1_ba1local_seed"
    #new_iid_file_g = "fashion_MNISTIID_size_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    
    #new_ci_file_l = "fashion_MNISTci_size_comp_FNN_lr0.5_lepo1_ba1local_seed"
    #new_ci_file_g = "fashion_MNISTci_size_comp_FNN_lr0.5_lepo1_ba1_global_seed"
    

    f_g = np.zeros((4,100))
    f_l = np.zeros((4,100))
    f_ci_g = np.zeros((4,100))
    f_ci_l = np.zeros((4,100))

    f_ci_new_g = np.zeros((4,100))
    f_iid_new_g = np.zeros((4,100))

    f_ci_new_l = np.zeros((4,10,100))
    f_iid_new_l = np.zeros((4,10,100))

    for i in range(4):
        f_g[i,:] = np.load(path + f_file_g + str(i) + ".npy")
        f_l[i,:] = np.mean(np.load(path + f_file_l + str(i) + ".npy"),axis=0)
        f_ci_g[i,:] = np.load(path + f_ci_file_g + str(i) + ".npy")
        f_ci_l[i,:] = np.mean(np.load(path + f_ci_file_l + str(i) + ".npy"),axis=0)
        
        #f_ci_new_g[i,:] = np.load(new_path + new_ci_file_g + str(i) + ".npy")
        #f_ci_new_l[i,:,:] = np.load(new_path + new_ci_file_l + str(i) + ".npy")
        
        #f_iid_new_g[i,:] = np.load(new_path + new_iid_file_g + str(i) + ".npy")
        #f_iid_new_l[i,:,:] = np.load(new_path + new_iid_file_l + str(i) + ".npy")

        
    #to_plot_FNN = [f_g, f_ci_g, cent_f]
    to_plot_FNN = {
        "iid" : f_g,
        "ci"  : f_ci_g,
        #"iid new" : f_iid_new_g,
        #"iid new l" : f_iid_new_l,
        #"ci new"  : f_ci_new_g,
        #"ci new l": f_ci_new_l,
        "central" : cent_f
    }
    return to_plot_FNN

def load_f_CNN():
    prefix = "CNN/fashion MNIST/"

    cnn_central_f_5e2 = np.load(prefix + "central_cnn_fashion_MNIST_lr5e-2_ba10_500.npy").T
    cnn_central_f_5e3 = np.load(prefix + "central_cnn_fashion_MNIST_lr5e-3_ba10_500.npy").T

    f_g_file = "fashion_MNISTIID_no_comp_CNN_lr0.05_lepo1_ba10_global_seed"
    f_l_file = "fashion_MNISTIID_no_comp_CNN_lr0.05_lepo1_ba10local_seed"

    f_ci_g_file = "../fashion MNIST CI/fashion_MNISTci_size_comp_CNN_lr0.05_lepo1_ba10_global_seed"
    f_ci_l_file = "../fashion MNIST CI/fashion_MNISTci_size_comp_CNN_lr0.05_lepo1_ba10local_seed"

    f_pmap_file = "fashion_MNISTIID_no_comp_CNN_lr0.05_lepo1_ba10prevmap_seed"
    f_nmap_file = "fashion_MNISTIID_no_comp_CNN_lr0.05_lepo1_ba10newmap_seed"

    cnn_f_g = np.zeros((500,4))
    cnn_f_l = np.zeros((10,500,4))
    cnn_f_ci_g = np.zeros((100,4))
    cnn_f_ci_l = np.zeros((10,100,4))

    f_pmap = []
    f_nmap = []

    for i in range(4):
        cnn_f_g[:,i] = np.load(prefix + f_g_file + str(i) + ".npy")
        cnn_f_l[:,:,i] = np.load(prefix + f_l_file + str(i) + ".npy")

        cnn_f_ci_g[:,i] = np.load(prefix + f_ci_g_file + str(i) + ".npy")
        cnn_f_ci_l[:,:,i] = np.load(prefix + f_ci_l_file + str(i) + ".npy")
        
        f_pmap.append(np.load(prefix + f_pmap_file + str(i) + ".npy"))
        f_nmap.append(np.load(prefix + f_nmap_file + str(i) + ".npy"))
        
    to_plot_CNN = [cnn_f_g, cnn_central_f_5e2, cnn_central_f_5e3]
    to_plot_CNN = {
        "iid" : cnn_f_g,
        "ci"  : cnn_f_ci_g,
        "cent 5e2": cnn_central_f_5e2,
        "cent 5e3": cnn_central_f_5e3
    }
    return to_plot_CNN

####### A2 #######

def load_A2_LR():
    pass

def load_A2_SVM():
    path = "SVM/new/A2/"
    central_A2 = np.load(path + "cent_SVM_A2_lr5e-05.npy").T
    central_A2_5e6 = np.load(path + "cent_SVM_A2_lr5e-06.npy").T

    global_iid_file = "IID/A2_PCAIID_no_comp_SVM_lr5e-05_lepo1_ba1_global_seed"
    local_iid_file = "IID/A2_PCAIID_no_comp_SVM_lr5e-05_lepo1_ba1_local_seed"

    global_ci_nc_file = "CI/A2_PCAci_no_comp_SVM_lr5e-05_lepo1_ba1_global_seed"
    local_ci_nc_file = "CI/A2_PCAci_no_comp_SVM_lr5e-05_lepo1_ba1_local_seed"
    global_scaf_file = "CI/A2_PCAci_scaf_SVM_lr5e-05_lepo1_ba1_global_seed" 
    local_scaf_file = "CI/A2_PCAci_scaf_SVM_lr5e-05_lepo1_ba1_local_seed"

    global_si_nc_file = "SI/A2_PCAsi_no_comp_SVM_lr5e-05_lepo1_ba1_global_seed"
    local_si_nc_file = "SI/A2_PCAsi_no_comp_SVM_lr5e-05_lepo1_ba1_local_seed"
    global_si_wc_file = "SI/A2_PCAsi_size_comp_SVM_lr5e-05_lepo1_ba1_global_seed"
    local_si_wc_file = "SI/A2_PCAsi_size_comp_SVM_lr5e-05_lepo1_ba1_global_seed"

    iid_newmap_file = "IID/A2_PCAIID_no_comp_SVM_lr5e-05_lepo1_ba1nc_newmap_seed"
    iid_prevmap_file = "IID/A2_PCAIID_no_comp_SVM_lr5e-05_lepo1_ba1prevmap_seed"

    ci_nc_newmap_file = "CI/A2_PCAci_no_comp_SVM_lr5e-05_lepo1_ba1nc_newmap_seed"
    ci_nc_prevmap_file = "CI/A2_PCAci_no_comp_SVM_lr5e-05_lepo1_ba1prevmap_seed"
    ci_scaf_newmap_file = "CI/A2_PCAci_scaf_SVM_lr5e-05_lepo1_ba1nc_newmap_seed"
    ci_scaf_prevmap_file = "CI/A2_PCAci_scaf_SVM_lr5e-05_lepo1_ba1prevmap_seed"

    si_nc_newmap_file = "SI/A2_PCAsi_no_comp_SVM_lr5e-05_lepo1_ba1nc_newmap_seed"
    si_nc_prevmap_file = "SI/A2_PCAsi_no_comp_SVM_lr5e-05_lepo1_ba1prevmap_seed"
    si_wc_newmap_file = "SI/A2_PCAsi_size_comp_SVM_lr5e-05_lepo1_ba1nc_newmap_seed"
    si_wc_prevmap_file = "SI/A2_PCAsi_size_comp_SVM_lr5e-05_lepo1_ba1prevmap_seed"

    svm_A2_iid_g = np.zeros((100,4))
    svm_A2_iid_l = np.zeros((10,100,4))
    svm_A2_ci_nc_g = np.zeros((100,4))
    svm_A2_ci_nc_l = np.zeros((10,100,4))
    svm_A2_scaf_g = np.zeros((100,4))
    svm_A2_scaf_l = np.zeros((10,100,4))
    svm_A2_si_nc_g = np.zeros((100,4))
    svm_A2_si_nc_l = np.zeros((10,100,4))
    svm_A2_si_wc_g = np.zeros((100,4))
    svm_A2_si_wc_l = np.zeros((10,100,4))


    for i in range(4):
        svm_A2_iid_g[:,i] = np.load(path + global_iid_file + str(i) +  ".npy")
        svm_A2_iid_l[:,:,i] = np.load(path + local_iid_file + str(i) +".npy")

        svm_A2_ci_nc_g[:,i] = np.load(path + global_ci_nc_file + str(i) +  ".npy")
        svm_A2_ci_nc_l[:,:,i] = np.load(path + local_ci_nc_file + str(i) +".npy")

        svm_A2_scaf_g[:,i] = np.load(path + global_scaf_file + str(i) +  ".npy")
        svm_A2_scaf_l[:,:,i] = np.load(path + local_scaf_file + str(i) +".npy")

        svm_A2_si_nc_g[:,i] = np.load(path + global_si_nc_file + str(i) +  ".npy")
        svm_A2_si_nc_l[:,:,i] = np.load(path + local_si_nc_file + str(i) +".npy")
        svm_A2_si_wc_g[:,i] = np.load(path + global_si_wc_file + str(i) +  ".npy")
        svm_A2_si_wc_l[:,:,i] = np.load(path + local_si_wc_file + str(i) +".npy")
        
    #to_plot_svm = [svm_A2_iid_g, svm_A2_ci_nc_g, svm_A2_si_wc_g, central_A2, central_A2_5e6]
    to_plot_svm = {

    "iid"       : svm_A2_iid_g,
    "ci"        : svm_A2_ci_nc_g,
    "si"        : svm_A2_si_wc_g,
    "central"   : central_A2,
    "central 5e6" :  central_A2_5e6 
    }
    return to_plot_svm

def load_A2_DT():
    pass

def load_A2_FNN():
    pass

def load_A2_CNN():
    pass

###### 3node ######

def load_3node_LR():
    path = "LR/new/3node/"

    central_3n = np.load(path + "3node_LR_cent.npy")
    central_3n_1e5 = np.load(path + "3node_LR_cent_1e5.npy" )
    file_3n_g = "3nodeIID_size_comp_LR_lr0.0001_lepo1_ba1_global_seed"
    file_3n_l = "3nodeIID_size_comp_LR_lr0.0001_lepo1_ba1_local_seed"

    node3_global = np.zeros((4,200))
    node3_local = np.zeros((4,200))

    for i in range(4):
        node3_global[i,:] = np.load(path + file_3n_g + str(i) + ".npy")
        node3_local[i,:] = np.mean(np.load(path + file_3n_l + str(i) + ".npy"), axis=0)

    #to_plot_lr = [node3_global, central_3n, central_3n_1e5]
    to_plot_lr = {

    "iid"       : node3_global,
    "central"   : central_3n,
    "central 1e5" :  central_3n_1e5 
    }
    return to_plot_lr

def load_3node_SVM():
    pass

def load_3node_DT():
    pass

def load_3node_FNN():
    prefix = "FNN/RMA/3node/"

    central_3n = np.load(prefix + "cent_3n_fnn_5e3.npy")
    central_redemp = np.load(prefix + "redemption_arc/cent_3node_FNN_redemp_lr5e-3.npy")
    central_redemp_lr = np.load(prefix + "redemption_arc/cent_3node_redemp_lower_lr.npy")


    g_3n_iid_file = "3nodeIID_size_comp_FNN_lr5e-05_lepo1_ba1_global_seed"
    l_3n_iid_file = "3nodeIID_size_comp_FNN_lr5e-05_lepo1_ba1local_seed"

    g_3n_ci_file = "3nodeCI_size_comp_FNN_lr0.005_lepo1_ba1_global_seed"
    l_3n_ci_file = "3nodeCI_size_comp_FNN_lr0.005_lepo1_ba1local_seed"

    redemption_g_file = "redemption_arc/3nodeci_size_comp_FNN_lr0.005_lepo1_ba1_global_seed"
    redemption_l_file = "redemption_arc/3nodeci_size_comp_FNN_lr0.005_lepo1_ba1local_seed"

    g_3n_ci = np.zeros((100,1))
    l_3n_ci = np.zeros((3,100,1))
    g_3n_iid = np.zeros((100,1))
    l_3n_iid = np.zeros((3,100,1))

    redemption_g = np.zeros((4,100))
    redemption_l = np.zeros((4,3,100))


    for i in range(4):
        redemption_g[i,:] = np.load(prefix + redemption_g_file + str(i) + ".npy")
        redemption_l[i,:,:] = np.load(prefix + redemption_l_file + str(i) + ".npy")

    for i in range(1):
        g_3n_ci[:,i] = np.load(prefix + g_3n_ci_file + str(i) + ".npy")
        l_3n_ci[:,:,i] = np.load(prefix + l_3n_ci_file + str(i) + ".npy")
            
        g_3n_iid[:,i] = np.load(prefix + g_3n_iid_file + str(i) + ".npy")
        l_3n_iid[:,:,i] = np.load(prefix + l_3n_iid_file + str(i) + ".npy")
    
    #to_plot_FNN  = [g_3n_ci, g_3n_iid]
    to_plot_FNN = {

    "iid"       : g_3n_iid,
    "ci"        : g_3n_ci,
    "redemp g"  : redemption_g,
    "redemp l"  : redemption_l,
    "ci local"  : l_3n_ci,
    "central"   : central_3n,
    "redemp central": central_redemp,
    "redemp central lr" : central_redemp_lr
    }
    return to_plot_FNN

def load_3node_CNN():
    pass


###### 2node ######

def load_2node_FNN():
    prefix = "FNN/2node/"

    cent_2n = np.load(prefix + "cent_2node_FNN.npy")

    g_2n_file = "2nodeIID_size_comp_FNN_lr0.005_lepo1_ba1_global_seed"
    l_2n_file = "2nodeIID_size_comp_FNN_lr0.005_lepo1_ba1local_seed"

    g_2n = np.zeros((100,4))
    l_2n = np.zeros((2,100,4))

    for i in range(4):
        g_2n[:,i] = np.load(prefix + g_2n_file + str(i) + ".npy")
        l_2n[:,:,i] = np.load(prefix + l_2n_file + str(i) + ".npy")

    #to_plot = [g_2n, cent_2n]
    to_plot_FNN = {

    "iid"       : g_2n,
    "central"   : cent_2n,
    }
    return to_plot_FNN