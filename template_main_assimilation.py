# Modules to interface with system
import sys
import os
import glob

# Modules to manage IOfiles
import csv
import fileinput

# Modules for maths/processing
import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import interp2d
from astropy.convolution import convolve_fft, Gaussian2DKernel, Tophat2DKernel, TrapezoidDisk2DKernel, convolve

# Modules for visualization
import matplotlib.pyplot as plt

# Modules to manage IOfiles
from struct import *

# Modules to manage time
from datetime import datetime
from datetime import date
import time

# Modules for statistical inference tool
import openturns as ot
from openturns.viewer import View

# Modules for sampling methods
from smt.sampling_methods import LHS

#****************************************************************************************#
#****************************************************************************************#
### Some useful functions
#****************************************************************************************#
#****************************************************************************************#

def fitKriging(coordinates, observations, covarianceModel, basis, lower, upper):
    '''
    Fit the parameters of a Kriging metamodel.
    '''
    # Define the Kriging algorithm.
    algo = ot.KrigingAlgorithm(
        coordinates, observations, covarianceModel, basis)

    # Set the optimization bounds for the scale parameter to sensible values
    # given the data set.
    scale_dimension = covarianceModel.getScale().getDimension()
    algo.setOptimizationBounds(ot.Interval([lower] * scale_dimension,
                                           [upper] * scale_dimension))

    # Run the Kriging algorithm and extract the fitted surrogate model.
    algo.run()
    krigingResult = algo.getResult()
    krigingMetamodel = krigingResult.getMetaModel()
    return krigingResult, krigingMetamodel
#****************************************************************************************#

def plotKrigingPredictions(krigingMetamodel, MyMeshBox, ny, nx):
    '''
    Plot the predictions of a Kriging metamodel.
    '''

    # Predict
    vertices = myMeshBox.getVertices()
    predictions = krigingMetamodel(vertices)

    # Format for plot
    X = np.array(vertices[:, 0]).reshape((ny, nx))
    Y = np.array(vertices[:, 1]).reshape((ny, nx))
    predictions_array = np.array(predictions).reshape((ny, nx))

    # Plot
    plt.figure()
    plt.pcolormesh(X, Y, predictions_array, shading='auto')
    plt.colorbar()
    plt.show()
    return
#****************************************************************************************#

# Python program to plot on 2D mesh 
def plotInterpolatedField(predictions, vertices, ny, nx):
    '''
    Plot the predictions.
    '''

    # Format for plot
    X = np.array(vertices[:, 0]).reshape((ny, nx))
    Y = np.array(vertices[:, 1]).reshape((ny, nx))
    predictions_array = np.array(predictions).reshape((ny, nx))

    # Plot
    plt.figure()
    plt.pcolormesh(X, Y, predictions_array, shading='auto')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.show()
    return
#****************************************************************************************#

# Python program to random sampling 
def latin_hypercube_sampling(alphalimits, N_ens):

  sampling = LHS(xlimits=alphalimits)
  alpha_ens = sampling(N_ens)
  
  return alpha_ens
#****************************************************************************************#

# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)
#****************************************************************************************#

def loadImageRaw(fname,res):
    import numpy as np
    with open(fname) as fid:
        T = np.fromfile(fid,dtype=np.float32)
        return np.reshape(T,res)
#****************************************************************************************#

# Check if a string exists in a file
def check_if_string_in_file(file_name, string_to_search):
    """ Check if any line in the file contains given string """
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            if string_to_search in line:
                return True
    return False
#****************************************************************************************#

# Search for multiple strings in a file and get lines containing string along with line numbers
def search_multiple_strings_in_file(file_name, list_of_strings):
    """Get line from the file along with line numbers, which contains any string from the list"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            line_number += 1
            # For each line, check if line contains any string from the list of strings
            for string_to_search in list_of_strings:
                if string_to_search in line:
                    # If any string is found in line, then append that line along with line number in list
                    list_of_results.append((string_to_search, line_number, line.rstrip()))
    # Return list of tuples containing matched string, line numbers and lines where string is found
    return list_of_results
#****************************************************************************************#

# Search for a string in file & get all lines containing the string along with line numbers
def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append((line_number, line.rstrip()))
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results
#****************************************************************************************#

def replacement(file, newline, matchLine):
   l=1
   for line in fileinput.input(file, inplace=1):
       oldline=line.strip("*")
       line=line.strip("*")
       if ( matchLine == l ) : 
          line = line.replace(oldline, newline + "\n")
       sys.stdout.write(line)
       l+=1
#****************************************************************************************#
# choice overAbsolute exponential or squared exponential
def C(s,t,l):
   return np.exp(-0.5*np.sqrt((s-t).dot((s-t)))/l)
   #return np.exp(-0.5*(s-t).dot((s-t))/l**2)

#****************************************************************************************#

def main():

# Some server adresses
  client_id="mbenali@ldmpe410h"
  server_cpu_id="mbenali@sator"
  server_cpu_id_spiro="mbenali@spiro03-clu"
  server_gpu_id="ybenali@republic03"

  rep_AD_results="/stck/mbenali"
  rep_AD_code="/tmp_user/sator/mbenali"
  rep_home_gpu="/home/ybenali/Documents"

  rep_code_gpu="/home/ybenali/Documents/Code_raytracer/Raytracer2"

# Create/check Project directory name
  rep_name = str(sys.argv[1])
  print(" Repertory name : "+rep_name)
  nb = 0
  if not rep_name :
    print(" Warning !!! : directory name to use/create is not specified ")
    while True :
      rep_name = str(nb)
      if ( os.path.isdir( rep_AD_results +"/Assimilation_dir_"+rep_name ) ):
         nb +=1
         continue
      break
    os.system(' ssh -t '+server_cpu_id+' "cp -r template_assimilation_results Assimilation_dir_'+rep_name+' ; cd '+rep_AD_code+' ; cp -r template_forecast forecast_dir_'+rep_name+' " ' )
    os.system('ssh -t '+server_gpu_id+' "cd '+rep_home_gpu+' ; mkdir Assimilation_dir_'+rep_name+' " ')
    print(" New directory is specified with name : Assimilation_dir_"+rep_name)
  else :  
    if not os.path.isdir(rep_AD_results+"/Assimilation_dir_"+rep_name) :
      os.system(' ssh -t '+server_cpu_id+' "cp -r template_assimilation_results Assimilation_dir_'+rep_name+' ; cd '+rep_AD_code+' ; cp -r template_forecast forecast_dir_'+rep_name+' " ' )
      os.system('ssh -t '+server_gpu_id+' "cd '+rep_home_gpu+' ; mkdir Assimilation_dir_'+rep_name+' " ')
    
# Assign some directories
  rep_AD_results = rep_AD_results +"/Assimilation_dir_"+rep_name
  rep_AD_code = rep_AD_code +"/forecast_dir_"+rep_name
  rep_home_gpu = rep_home_gpu +"/Assimilation_dir_"+rep_name
  rootdir = rep_AD_results+"/inputs"
  rep_CHARME_BOS = rep_AD_results + "/CHARME_BOS"
  rep_input_calib = rep_CHARME_BOS + "/input/Calib"
  dir_iter ="assim_iteration_"
      
# Some prefixes of files
  prefix_in_bos_filename="bos_input_volume_"
  prefix_out_bos_filename="dev"

  prefix_out_cedre_filename="coords_state_and_ens_"
  prefix_out_cedre_pre_in_bos_filename="bos_coords_and_input_volume_"

# Some path to scripts
  path_to_tool='/stck/mbenali/tools/paraview/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/bin/pvpython'

  small = 1e-15
   
# Setting for ensemble KF
  N_ens = 40
  inv_Nens = 1. / N_ens

  MRMSE_STATE=[]

# Number of modes to keep in KL decomposition
  NbModes = 30 
  
# Setting for filters called by PDAF
  rms_obs = 1e-3
  forget = 1.0

# Ranges of sampled parameters
  # Considering k-w-sst coefficients
  p_names = 'Cmu,prtt,sigma_k2,sigma_w2,beta2,gamma2'
  p_values_word = '0.0900,0.900,1,0.8560,0.08280,0.440355'
  p_values =  [float(j) for j in p_values_word.split(",")]

# Columns collecting the extracted features
  all_columns_to_keep = ["Grad~K~:0","Grad~K~:1","Grad~K~:2","Grad~K~_MV","Grad~T~:0","Grad~T~:1","Grad~T~:2","Grad~T~_MV"]
  all_uncertain_columns = ["deltaRho","deltaRhoU","deltaRhoV","deltaRhoW","deltaRhoE","deltaRhoZ1","deltaRhoZ2"]

  var_name = str(sys.argv[2].split(",")[0])
  eq_name = str(sys.argv[2].split(",")[1])

  weight_name = str(sys.argv[3].split(",")[1])
  if weight_name == "none" or weight_name == "None":
    columns_to_keep = ["Grad~"+var_name+"~_MV"]
  else :
    columns_to_keep = ["Grad~"+var_name+"~_MV", "Grad~"+weight_name+"~_MV"]

  uncertain_columns = ["deltaRho"+eq_name]
  print("Entree 1 : "+str(var_name) )
  print("Entree 2 : "+str(eq_name) )
  print("Entree 3 : "+str(weight_name) )

# Domain setting for uncertainty
  D = 0.06
  d = 2
  min_xtp, max_xtp= 0.0, 2.95
  min_ytp, max_ytp= 0.0, 4.0
  res_y = 90
  res_x = int(res_y * (max_xtp-min_xtp)/(max_ytp-min_ytp))
  l_caract = str(sys.argv[4].split(",")[1])
  scale = str(sys.argv[5].split(",")[1])
  Eps = float(scale)
  h = float(l_caract)
  print("Entree 4 : "+str(Eps) )
  print("Entree 5 : "+str(h) )
  size_kernel= 3
  nlayers = 1

  isExplicitkernel = False
  kernelconstruction = str(sys.argv[6].split(",")[1])
  print("Entree 6 : "+str(kernelconstruction) )
  if kernelconstruction == "explicit" :
    isExplicitkernel = True

# Observation volume structure given through rotational extrusion
  vol_resolution = ["R",57, 151, 151]
  vol_scale = ["S",0.17, 0.45, 0.45]
  # the vol_edge_origine is not the actual origine, it actually serves as an edge point 
  vol_edge_origin = ["O",0.005, -0.225, -0.225]
  vol_structure = []

  args = [
		      str(''.join([','+str(i) for i in vol_edge_origin])[1:]),
		      str(''.join([','+str(i) for i in vol_scale])[1:]),
		      str(''.join([','+str(i) for i in vol_resolution])[1:])
		   ]

  vol_origin = list(np.array(vol_edge_origin[1:]) + np.array(vol_scale[1:])/2)
  vol_structure.extend(vol_origin)    
  vol_structure.extend(vol_resolution[1:])
	    
  voxel_size = vol_scale[1]/vol_resolution[1]
  vol_structure.append(voxel_size)

# Setting for BOS
  # real CCD resolution
  #n_row = 2058
  #n_col = 2456
  # synthetic CCD resolution
  n_row = 500
  n_col = 500
  l_x = 18
  l_y = 19
  tpx = - vol_origin[0]
  fx = 1.200000e+03 * n_row / 500
  fy = 1.200000e+03 * n_col / 500
  cx = n_row / 2 - 0.5
  cy = n_col / 2 - 0.5
  inputs_Calib = ["tpx","fx","fy","cx","cy"]
  inputs_Calib_values = [tpx,fx,fy,cx,cy]
# Check for iteration
  it = 0
  while True :
     if ( os.path.isfile( rep_AD_results+'/'+dir_iter+str(it)+'/outputs/log_pdaf_'+str(N_ens) ) ):
       it += 1
       continue
     else :
       break
       
  time.sleep(2)

# Some characteristic values (problem specific)
  Uref=30
  kref = Uref**2
  Tref= 298.15
  # turbulence lenght scale in pipe
  l = 0.07*D 
  Omegaref = 0.09**(0.75) * Uref/l
  muref = kref / Omegaref
  Rho_ref=1.18

  while it < 15:

    DAY = date.today().day
    MONTH = date.today().month
    YEAR = date.today().year
    now = datetime.now().time() # time object
    current_time = now.strftime("%H:%M:%S")
    print('	')
    print('********************************************************************** ')
    print('************ Start iteration '+str(it)+' of assimilation process ************ ')
    print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
    print('********************************************************************** ')
    print('	')
    
    if not os.path.isdir(rep_AD_results+'/'+dir_iter+str(it)) :
      os.system('cd '+rep_AD_results+' ; mkdir '+dir_iter+str(it) )
      os.system('cd '+str( os.path.join(rep_AD_results, dir_iter+str(it))+' ; mkdir inputs ; mkdir outputs ' ))   

    print('	')
    print('********************************************************************* ')
    print('****************** Proceed for model forcast ************************ ')
    print('********************************************************************* ')
    print('	')
       
    if ( os.path.isfile( rootdir+'/log_forcast_state_mean' ) and check_if_string_in_file(rootdir+'/log_forcast_state_mean', "FIN DE L'ECRITURE") ):
	    print('	')
	    print("********************** Forcast step already done ***************************** ")
	    print('	')
    else :
            if it == 0 :
              if ( os.path.isfile( rootdir+'/log_pre_forcast_state_mean' ) and check_if_string_in_file(rootdir+'/log_pre_forcast_state_mean', "FIN DE L'ECRITURE") ):
                print('	')
                print("**************** Pre-forcast step is already done ******************** ")
                print('	')
              else :
                print('	')
                print("**************** Pre-forcast step to extract flow features ******************** ")
                print('	')
                os.system('ssh -t '+server_cpu_id+' "cd '+rep_AD_code+' ; bash model_pre_forcast_baseline.sh -o '+rep_AD_results+' -d '+rep_AD_code+' " ')
	     
                while  True :
                  if ( os.path.isfile( rootdir+'/log_pre_forcast_state_mean' ) and check_if_string_in_file(rootdir+'/log_pre_forcast_state_mean', "FIN DE L'ECRITURE") ):
                    break
                  time.sleep(10)
                  continue
                
                if ( os.path.isfile( rootdir+'/log_extraction_mean' ) and check_if_string_in_file(rootdir+'/log_extraction_mean', "END") ):
                    print('	')
                    print("******************* Feature extraction already done on baseline ***************************** ")
                    print('	')
                else :
                    print('******************* Perform feature extraction on baseline ********************* ')
                    print('	')
                    script='feature_extraction.py'
                    #os.system('ssh -t '+server_cpu_id_spiro+' "cd '+rep_AD_results+' ;'+path_to_tool+' '+script+' '+rootdir+' mean > '+rootdir+'/log_extraction_mean " ')
                    os.system(' cd '+rep_AD_results+' ;'+path_to_tool+' '+script+' '+rootdir+' mean > '+rootdir+'/log_extraction_mean  ')
                    
                    while  True :
                      if ( os.path.isfile( rootdir+'/log_extraction_mean' ) and check_if_string_in_file(rootdir+'/log_extraction_mean', "END") ):
                        break
                      time.sleep(5)
                      continue
 
              DAY = date.today().day
              MONTH = date.today().month
              YEAR = date.today().year
              now = datetime.now().time() # time object
              current_time = now.strftime("%H:%M:%S")
              print('	') 
              print('******************* Perform KL-decomposition ********************* ')
              print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
              print('	') 
              os.chdir(rootdir)
              dummy_file = 'coords_features_and_ens_mean.csv'
              columns_ = ["Points:0","Points:1","Points:2"]
              data = pd.read_csv(dummy_file, usecols = columns_, skiprows= 0, sep=',')
              coords_x = list(data[columns_[0]])
              coords_y = list(data[columns_[1]])

              # Prepare Fields and set up the Mesh, ProcessSample to the KL decomposition
              # Step 1 : Define Mesh as it is necessary for all the uncertainty decomposition procedure, i.e. ProcessSample has to be projected into it. 
              # Define the vertices of the mesh: Get them from computational mesh for instance
              '''
              coordinates = list(zip(np.array(coords_x),np.array(coords_y)))
              coordinates = [list(v) for v in coordinates]
              
              # Define the simplices that prescribe a topology (connections) of the mesh :(Needs more work). 
              #simplices = [[0, 1, 2], [1, 2, 3]]
              meshd = ot.Mesh(coordinates, simplices)
              '''

              # Or create from msh file (freefem) : Not working and needs instruction guides
              '''
              fileName=rootdir+'/JET.msh'
              meshd.ImportFromMSHFile(fileName)
              '''

              # Regular meshgrid defintion (Box domain)

              myIndices = [res_x-1, res_y-1]
              myMesher = ot.IntervalMesher(myIndices)
              lowerBound = [0.0, 0.0]
              upperBound = [max_xtp*D, max_ytp*D]
              myInterval = ot.Interval(lowerBound, upperBound)
              meshd = myMesher.build(myInterval)
              vertices = meshd.getVertices()  
              v_x = np.array(vertices[:, 0])
              v_y = np.array(vertices[:, 1]) 
              size_grid = meshd.getVerticesNumber()          
              print(' Nombre de points pour Cilepi : '+str(size_grid))
              #X = np.array(vertices[:, 0]).reshape((res_y, res_x))
              #Y = np.array(vertices[:, 1]).reshape((res_y, res_x))

              '''
              # Create a graph and draw the mesh
              aGraph = meshd.draw()
              View(aGraph).show()
              plt.gca().set_aspect('equal')
              plt.show()
              '''

              # Export mesh to VTK (could be  worked for generalization)
              # meshd.exportToVTKFile(rootdir+'/myVTKFile.vtk')
              meshd.computeWeights()

              # save coords as "nuage de point" input to cilepi
              grid_dict = { 'x': [float(i[0]) for i in v_x], 'y': [float(i[0]) for i in v_y], 'z': [0*float(i[0]) for i in v_x] }
              grid_to_cilepi = pd.DataFrame.from_dict(grid_dict)
              grid_to_cilepi.to_csv(rootdir+'/coords_space_for_cilepi.csv', index=False, header=False, sep=' ')

              print(">>>> Mesh created succesfully in OpenTurns")

              # Iinitialize uncertainty files with zero columns
              uncertain_dict = { j:[] for j in all_uncertain_columns }
              for col in range(len(all_uncertain_columns)):
                uncertain = np.ones(size_grid)*0.
                uncertain_dict[ all_uncertain_columns[col] ] = list(uncertain)     
              init_uncertain = pd.DataFrame.from_dict(uncertain_dict)
           
              for m in range(N_ens):
                member = m+1
                init_uncertain.to_csv(rootdir+'/uncertain_'+'{:02d}'.format(member)+'_cilepi.csv', index=False, header=False, sep=' ')
              init_uncertain.to_csv(rootdir+'/uncertain_mean_cilepi.csv', index=False, header=False, sep=' ')

              ## scaling factor for the added uncertainties (free parameter)
              if len(columns_to_keep) == 1:
                Scaling=Eps*Rho_ref*Uref
              else : # if mutiplied by grad K
                Scaling=Eps*Rho_ref*D/Uref

              for col1 in range(len(uncertain_columns)):
                  print(">>>> Get sample features based on the field : "+columns_to_keep[col1])

                  SampleFields = ot.ProcessSample(meshd, N_ens, 1)

                  # Define kernel for the convolution
                  kernel = Gaussian2DKernel(size_kernel)
                  #kernel = RickerWavelet2DKernel(10)
                  #kernel = Tophat2DKernel(9, mode='linear_interp')
                  #kernel = TrapezoidDisk2DKernel(1, slope=0.2)
                  
                  mu_X = np.zeros((size_grid,))
                  threshold = 1e-8
                  # For large problems
                  ot.ResourceMap.GetBoolKeys()
                  ot.ResourceMap.SetAsBool('KarhunenLoeveSVDAlgorithm-UseRandomSVD', True)

                  # construct directly the covariance matrix => straighforward creation of Gaussian process
                  filename = 'coords_features_and_ens_mean.csv'
                  data = pd.read_csv(filename, usecols = columns_to_keep, skiprows= 0, sep=',')
                  np_Values = np.zeros((len(data[columns_to_keep[col1]]),))
                  np_Values = np.array(data[columns_to_keep[col1]])*Scaling

                  if len(columns_to_keep) != 1 :
                    for col2 in range(1,len(columns_to_keep)):
                      np_Values *= np.array(data[columns_to_keep[col2]])

                  Values = list(np_Values)

                  ValuesGrid = scipy.interpolate.griddata((coords_x, coords_y), Values , (v_x, v_y), method='linear', fill_value = 0.0)
                  SeqValues = ValuesGrid.flatten()
                  #title = 'Baseline feature ('+columns_to_keep[col1]+')'
                  #plotInterpolatedField(SeqValues, vertices, res_y, res_x,title)

                  nlayers = 1
                  for l in range(nlayers):
                      Image= np.array(SeqValues).reshape((res_y, res_x))
                      #newImage = convolve_fft(Image, kernel)
                      newImage = convolve(Image, kernel)
                      SeqValues = newImage.flatten()

                  #title = 'Smooth Baseline feature ('+columns_to_keep[col1]+')'
                  #plotInterpolatedField(SeqValues, vertices, res_y, res_x,title)
                  mu_X = SeqValues 
                
                  if isExplicitkernel :###### Explicitly construct the covariance kernel from a matrix, suitable for low-medium dimension (up to around 5000 grid cells is ok)
                    myCovariance = ot.CovarianceMatrix(meshd.getVerticesNumber())

                    now = datetime.now().time() # time object
                    current_time = now.strftime("%H:%M:%S")
                    print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
                    print(" construct covariance matrix ") 
                    for k in range(meshd.getVerticesNumber()):
                      t = meshd.getVertices()[k]
                      for l in range(k + 1):
                        s = meshd.getVertices()[l]
                        myCovariance[k, l] = mu_X[k]*mu_X[l]*C(s, t, h)
                    myCovarianceModel = ot.UserDefinedCovarianceModel(meshd, myCovariance)
                  
                    now = datetime.now().time() # time object
                    current_time = now.strftime("%H:%M:%S")
                    print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
 
                    print(" sample a gaussian process ")
                    process = ot.GaussianProcess(myCovarianceModel, meshd)
                    GaussSamples = process.getSample(N_ens)
                  else : #### Create a Gaussian process with a canonical model for the covariance kernel and then scaled according to the feature field, suitable for higher-dimension as the process being sampled 
                  
                    #myCovarianceModel= ot.IsotropicCovarianceModel(ot.SquaredExponential([h]), d)
                    myCovarianceModel= ot.IsotropicCovarianceModel(ot.AbsoluteExponential([h]), d)
 
                    print(" sample a gaussian process ")
                    process = ot.GaussianProcess(myCovarianceModel, meshd)
                    GaussSamples = process.getSample(N_ens)
                  
                    print(" rescale process samples ")
                    # Collect sampled features to model the process
                    for m in range(N_ens):
                      print("  sample :"+str(m+1))
                      tmp_SeqValues = np.array([ v[0] for v in GaussSamples.getField(m).getValues() ])
                      SeqValues = np.multiply(tmp_SeqValues,mu_X)
                      SeqValues = [ [v] for v in SeqValues ]
                      Fieldm = ot.Field(meshd, SeqValues)
                      GaussSamples.setField(Fieldm,m)
                  
                  print(">>>> Compute the SVD decomposition of the process")

                  algo_gauss = ot.KarhunenLoeveSVDAlgorithm(GaussSamples, threshold)
                  algo_gauss.run()
                  result = algo_gauss.getResult()

                  # Get KL modes scaled by their corresponding eigenvalues
                  phi = result.getScaledModesAsProcessSample()
                  # Get KL eignevectors   
                  #phi = result.getModesAsProcessSample()
                  # Get KL eigenvalues
                  #lambda_ = result.getEigenvalues()
                  
                  NbModes = phi.getSize()
                  for i in range(NbModes):
                    #plotInterpolatedField(phi.getField(i).getValues(), vertices, res_y, res_x)
                    np.savetxt(rootdir+'/phi_'+uncertain_columns[col1]+'_'+'{:02d}'.format(i)+'.csv', phi.getField(i).getValues())

                  print('	')
                  print('************** Save uncertain samples into cilepi readable input ************** ')
                  print('	')

                  #### Generate quasi-random parameter weights
                  mean_ens = np.zeros((NbModes,1))
                  ens =  np.random.normal(0., 1.0, size=(N_ens, NbModes))
                  #print(ens)

                  ##### Save weights and uncertainties of sample mean
                  np.savetxt(rootdir+'/weights_'+uncertain_columns[col1]+'_mean.csv', mean_ens)

                  size_grid = meshd.getVerticesNumber()
                  uncertain = np.zeros((size_grid,1))
                  for i in range(NbModes):
                   uncertain += np.array(mean_ens[i]*phi.getField(i).getValues()) 
                  
                  # write uncertainties to dataframe
                  data = pd.read_csv(rootdir+'/uncertain_mean_cilepi.csv', usecols=[i for i in range(len(all_uncertain_columns))], names=all_uncertain_columns, sep=' ')
                  for col2 in range(len(all_uncertain_columns)):
                       if ( uncertain_columns[col1] != all_uncertain_columns[col2] ):
                          continue
                       else :
                          data[all_uncertain_columns[col2]] = uncertain
                    
                  data.to_csv(rootdir+'/uncertain_mean_cilepi.csv', index=False, header=False, sep=' ')

                  '''
                  uncertain = np.ones(size_grid)*0.
                  for i in range(NbModes):
                     uncertain += mean_ens[i]*phi.getField(i).getValues()
                  if col1 == 0 :  
                      np.savetxt(rootdir+'/uncertain_mean_cilepi.csv', uncertain)
                  else :
                      with open(rootdir+'/uncertain_mean_cilepi.csv', 'r') as fi, open(rootdir+'/uncertain_mean_cilepi.csv', 'a+') as fo :
                        lines = [[i.strip() for i in line.strip().split(' ')] for line in fi.readlines()]
                        new_lines = [line + [str(uncertain[i])] for i, line in enumerate(lines)]
                        for line in new_lines:
                          fo.write(' '.join(line) + '\n')
                  '''

                  ##### Save weights and uncertainties of each sample
                  for m in range(N_ens):
                    member = str(m+1)
                    print("     member : "+str(member))
                    np.savetxt(rootdir+'/weights_'+uncertain_columns[col1]+'_'+'{:02d}'.format(int(member))+'.csv', ens[m])
                    uncertain = np.zeros((size_grid,1))
                    for i in range(NbModes):
                      uncertain += np.array(ens[m,i]*phi.getField(i).getValues())

                    #plotInterpolatedField(uncertain, vertices, res_y, res_x)
                    # write uncertainties to dataframe
                    data = pd.read_csv(rootdir+'/uncertain_'+'{:02d}'.format(int(member))+'_cilepi.csv', usecols=[i for i in range(len(all_uncertain_columns))], names=all_uncertain_columns, sep=' ')
                    for col2 in range(len(all_uncertain_columns)):
                       if ( uncertain_columns[col1] != all_uncertain_columns[col2] ):
                          continue
                       else :
                          data[all_uncertain_columns[col2]] = uncertain
                    
                    data.to_csv(rootdir+'/uncertain_'+'{:02d}'.format(int(member))+'_cilepi.csv', index=False, header=False, sep=' ')

              
              print('	')
              print('************** Execute cilepi to generate uncertainty for CEDRE ************** ')
              print('	')
              for m in range(N_ens):
                member = str( m+1 )
                print("     member : "+str(member))
                if ( os.path.isfile( rootdir+'/execute_cilepi_'+member+'.log') and check_if_string_in_file( rootdir+'/execute_cilepi_'+member+'.log', "ECRITURE DU FICHIER TERMINEE") ):
                  continue
                else :
                  # Transfer uncertainties
                  os.system('ssh -t '+server_cpu_id+' "cp '+rootdir+'/uncertain_'+'{:02d}'.format(int(member))+'_cilepi.csv '+rep_AD_code+'/model_forcast_'+member+'/state_source_for_cilepi.csv" ')
                  # Transfert coordinates
                  os.system('ssh -t '+server_cpu_id+' "cp '+rootdir+'/coords_space_for_cilepi.csv '+rep_AD_code+'/model_forcast_'+member+'/. " ')
                  # Generate source file
                  os.system('ssh -t '+server_cpu_id+' " source /etc/profile.d/modules.sh ; module use /opt/tools/modules/compilation; module use /opt/tools/modules/librairies; module use /opt/tools/modules/logiciels; module use /opt/tools/modules/mpi; module use /opt/tools/modules/obsoletes; module use /opt/tools/modules/optimisation; module use /opt/tools/modules/socles; module use /tmp_user/sator/cedre/module/chaine_cedre; module unload intel/17.0.4; module unload impi/17; module load python/3.6.1; module load intel/18.0.3; module load impi/18; module load cedre/8.1.0.3/intel18_impi18; cd '+rep_AD_code+'/model_forcast_'+member+'/; ../model_forcast/_CILEPI_/fichier_1/cilepi -epicea > execute_cilepi.log 2>&1 ; mv execute_cilepi.log '+rootdir+'/execute_cilepi_'+member+'.log " ')
 
                  while  True :
                    if ( os.path.isfile( rootdir+'/execute_cilepi_'+member+'.log') and check_if_string_in_file( rootdir+'/execute_cilepi_'+member+'.log', "ECRITURE DU FICHIER TERMINEE") ):
                      break
                    time.sleep(1) 
                    continue

              print("     Sample mean ")
              if ( os.path.isfile( rootdir+'/execute_cilepi_mean.log') and check_if_string_in_file( rootdir+'/execute_cilepi_mean.log', "ECRITURE DU FICHIER TERMINEE") ): 
                time.sleep(1)
              else :
                # Transfer uncertainties
                os.system('ssh -t '+server_cpu_id+' "cp '+rootdir+'/uncertain_mean_cilepi.csv '+rep_AD_code+'/model_forcast_mean/state_source_for_cilepi.csv" ')
                # Transfert coordinates
                os.system('ssh -t '+server_cpu_id+' "cp '+rootdir+'/coords_space_for_cilepi.csv '+rep_AD_code+'/model_forcast_mean/. " ')
                # Generate source file
                os.system('ssh -t '+server_cpu_id+' " source /etc/profile.d/modules.sh ; module use /opt/tools/modules/compilation; module use /opt/tools/modules/librairies; module use /opt/tools/modules/logiciels; module use /opt/tools/modules/mpi; module use /opt/tools/modules/obsoletes; module use /opt/tools/modules/optimisation; module use /opt/tools/modules/socles; module use /tmp_user/sator/cedre/module/chaine_cedre; module unload intel/17.0.4; module unload impi/17; module load python/3.6.1; module load intel/18.0.3; module load impi/18; module load cedre/8.1.0.3/intel18_impi18; cd '+rep_AD_code+'/model_forcast_mean/; ../model_forcast/_CILEPI_/fichier_1/cilepi -epicea > execute_cilepi.log 2>&1 ; mv execute_cilepi.log '+rootdir+'/execute_cilepi_mean.log " ')
 
                while  True :
                  if ( os.path.isfile( rootdir+'/execute_cilepi_mean.log') and check_if_string_in_file( rootdir+'/execute_cilepi_mean.log', "ECRITURE DU FICHIER TERMINEE") ):
                      break
                  time.sleep(1) 
                  continue

            DAY = date.today().day
            MONTH = date.today().month
            YEAR = date.today().year
            now = datetime.now().time() # time object
            current_time = now.strftime("%H:%M:%S")
            print('	')
            print('************************* Submit jobs to CEDRE ********************** ')
            print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
            print('	')
            os.system('ssh -t '+server_cpu_id+' "cd '+rep_AD_code+' ; bash model_forcast_local_uncertain.sh -e '+str(N_ens)+' -o '+rep_AD_results+' -d '+rep_AD_code+' -i '+str(it)+' " ')
            print('	')
            print('************************* Wait for model forcast  ******************* ')
            print('	')

            while  True :
              if ( os.path.isfile( rootdir+'/log_forcast_state_mean' ) and check_if_string_in_file(rootdir+'/log_forcast_state_mean', "FIN DE L'ECRITURE") ):
                break
              time.sleep(10)
              continue

    if ( os.path.isfile( rootdir+'/log_extrusion_'+str(N_ens) ) and 
         check_if_string_in_file(rootdir+'/log_extrusion_'+str(N_ens), "END") ):
	    print('	')
	    print("********************** Extrusion already done ***************************** ")
	    print('	')
    else :
	    print('********** Perform rotational extrusion (from 2D axi) *************** ')
	    print('	')

	# Execute rotational extrusion from 2d Axi to 3D and than interpolate into a 3D cartesan cubical grid
	# save state coords and data and equivalently for the observed state into ASCII csv files
	    script='rotational_extrusion_and_data_extraction_2DAxi_to_3D.py'

	# configurate CONF.ini for the above volume structure
	    inputs_CONF_ini = ["origin_x","origin_y","origin_z","n_vox_x","n_vox_y","n_vox_z","voxel_size"]
	    matched_lines = search_multiple_strings_in_file(rep_AD_results+'/CHARME_BOS/CONF.ini', inputs_CONF_ini)
	    for elem in range(len(matched_lines)):
	      newLine= inputs_CONF_ini[elem] + " = " + str(vol_structure[elem])
	      replacement(rep_AD_results+"/CHARME_BOS/CONF.ini", newLine, matched_lines[elem][1])
	    print('volume structure is initialized in conf.ini ')
	   
	    for m in range(N_ens):
	      member = m+1
	      if ( os.path.isfile( rootdir+'/log_extrusion_'+str(member) ) and check_if_string_in_file(rootdir+'/log_extrusion_'+str(member), "END") ):
	         print('	')
	         print("********************** Extrusion already done on member "+str(member)+" ***************************** ")
	         print('	')
	      else :
	         #os.system('ssh -t '+server_cpu_id_spiro+' "cd '+rep_AD_results+' ;'+path_to_tool+' '+script+' '+rootdir+' '+str(member)+' '+args[0]+' '+args[1]+' '+args[2]+' > '+rootdir+'/log_extrusion_'+str(member)+' " ')
	         os.system(' cd '+rep_AD_results+' ;'+path_to_tool+' '+script+' '+rootdir+' '+str(member)+' '+args[0]+' '+args[1]+' '+args[2]+' > '+rootdir+'/log_extrusion_'+str(member))

	    while  True :
	       if ( os.path.isfile( rootdir+'/log_extrusion_'+str(N_ens) ) and 
		 check_if_string_in_file(rootdir+'/log_extrusion_'+str(N_ens), "END") ):
                 break
	       time.sleep(5)
	       continue

	    print('	')
	    print("********************** Forcast step done ***************************** ")
	    print('	')


#*************************************************************************************************#
#*************************************************************************************************#
   
    DAY = date.today().day
    MONTH = date.today().month
    YEAR = date.today().year
    now = datetime.now().time() # time object
    current_time = now.strftime("%H:%M:%S") 
    print('	')
    print('********************************************************************** ')
    print('************** Run observation operator via raytracing ************** ')
    print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
    print('********************************************************************** ')
    print('	')

    columns_to_keep = ["Points:0","Points:1","Points:2","Rho"]
    state_columns = ["Rho"]

    os.chdir(rootdir)

    for m in range(N_ens):
      member = str(m+1)

      if ( os.path.isfile( rootdir + '/bos_input_volume_'+member+'.dat' ) ):
        continue
      print("******************* Prepare density volume "+member+"**************************** ")
      print('	')
      filename = prefix_out_cedre_pre_in_bos_filename +member+'.csv'
      data = pd.read_csv(filename, usecols = columns_to_keep, skiprows= 0, sep=',')
      coords_x, coords_y, coords_z, state_ = [], [], [], []
      for col in range(len(state_columns)):
        coords_x.extend(data['Points:0'])
        coords_y.extend(data['Points:1'])
        coords_z.extend(data['Points:2'])
        state_.extend(data[state_columns[col]])
   
      fileout = open("bos_input_volume_"+member+".dat", "wb")
      fileout.write(pack(str(len(state_))+'f', *state_))
      fileout.close()

    n_camera = 0
    for root, subdirs, files in os.walk(rep_input_calib):
      for subdir in subdirs:
         n_camera = 0 
         print(" Camera conf from folder :"+str(subdir) )
         os.chdir(str(os.path.join(root, subdir)))
         for file in glob.glob("camera*"):
            matched_lines = search_multiple_strings_in_file(str(file), inputs_Calib)
            for elem in range(len(matched_lines)):
              newLine= inputs_Calib[elem] +" "+ str(inputs_Calib_values[elem])
              replacement(file, newLine, matched_lines[elem][1])
            n_camera += 1
            newLine= "x " + str(n_row)
            replacement(file, newLine, 18)
            newLine= "y " + str(n_col)
            replacement(file, newLine, 19)

    os.chdir(rootdir)
    isDev = False
    os.system('scp '+rep_AD_results+'/wait_routine.py '+server_gpu_id+':'+rep_home_gpu+'/. ')
    for m in range(N_ens):
     member = str(m+1) 
     if ( os.path.isfile( rootdir+'/Deviation_'+member+'/dev1x.dat' ) ):
	     print( "Deviations "+member+ "are there : "+ str( os.path.isfile(rootdir+'/Deviation_'+member+'/dev1x.dat') ))
	     isDev = True
     else :
	     filename = prefix_in_bos_filename+member+'.dat'
	#    make directory for each state to be observed
	     os.system('ssh -t '+server_gpu_id+' "cd '+rep_home_gpu+' ; rm -r BOS_'+member+' " ')
	     os.system('scp -r '+rep_AD_results+'/CHARME_BOS'+' '+server_gpu_id+':'+rep_home_gpu+'/BOS_'+member+' > log_scp_bos')

	#    copy each input density field into the created directory
	     os.system('scp '+rep_AD_results+'/inputs/'+filename+' '+server_gpu_id+':'+rep_home_gpu+'/BOS_'+member+'/input/Calcul_CFD/ > log_scp_input_vol_bos')

	#    run raytracing code on each member
	#    tranfert BOS outputs files to input folder for analysis step

	     print('	')
	     print("***************** Submit job for raytracing ************************** ")
	     print('	')
	     print('ssh -t '+server_gpu_id+' "cd '+rep_home_gpu+'/BOS_'+member+'/ ; python3.6 change_conf_ini.py '+member+';'+rep_code_gpu+' ./CONF.ini " ')

	     os.system('ssh -t '+server_gpu_id+' "cd '+rep_home_gpu+'/BOS_'+member+'/ ; python3.6 change_conf_ini.py '+member+';'+rep_code_gpu+' ./CONF.ini > log_raytracer " ')

	     isDev = False
	     print('	')
	     print('************** Wait for observation on state '+member+'*************** ')
	     print('	')
	
    if ( isDev ):
       print( "Deviations "+member+ "are there : "+ str( os.path.isfile(rootdir+'/Deviation_'+member+'/dev1x.dat') ))
    else :  
       os.system('ssh -t '+server_gpu_id+' "cd '+rep_home_gpu+' ; python3.6 wait_routine.py '+str(N_ens)+' '+server_cpu_id+' '+rootdir+' " ')

    print('	')
    print("************** Observation projection step done on all member forcast *********************** ")
    print('	')
   
    
    os.chdir(rootdir)
    if ( os.path.isfile( rootdir + '/log_extrusion_mean' ) and 
         check_if_string_in_file( rootdir + '/log_extrusion_mean', "END") ):
	    print('	')
	    print('**************** Mean forcast state is already extracted ***************** ')
	    print('	')
    else :
	    print('	')
	    print('********** Perform rotational extrusion (from 2D axi) and prepare volume *************** ')
	    print('	')

	    script='rotational_extrusion_and_data_extraction_2DAxi_to_3D.py'

	    #os.system('ssh -t '+server_cpu_id_spiro+' "cd '+rep_AD_results+' ;'+path_to_tool+' '+script+' '+rootdir+' mean '+args[0]+' '+args[1]+' '+args[2]+' >   '+rootdir+'/log_extrusion_mean " ')

	    os.system(' cd '+rep_AD_results+' ;'+path_to_tool+' '+script+' '+rootdir+' mean '+args[0]+' '+args[1]+' '+args[2]+' >   '+rootdir+'/log_extrusion_mean ')

	    while  True :
	       if ( os.path.isfile( rootdir+'/log_extrusion_mean')  and 
		 check_if_string_in_file(rootdir+'/log_extrusion_mean', "END") ):
                 break
	       time.sleep(5)
	       continue

    if ( os.path.isfile( rootdir+'/Deviation_mean/dev1x.dat' ) ): 
	    print('	')
	    print('**************** Mean forcast state is already observed ***************** ')
	    print('	')
    else :
	    columns_to_keep = ["Points:0","Points:1","Points:2","Rho"]
	    state_columns = ["Rho"]

	    filename = prefix_out_cedre_pre_in_bos_filename +'mean.csv'
	    data = pd.read_csv(filename, usecols = columns_to_keep, skiprows= 0, sep=',')
	    state_ = list(data[state_columns[0]])
	   
	    fileout = open("bos_input_volume_mean.dat", "wb")
	    fileout.write(pack(str(len(state_))+'f', *state_))
	    fileout.close()

	    filename = 'bos_input_volume_mean.dat'
	#    make directory for each state to be observed
	    os.system('ssh -t '+server_gpu_id+' "cd '+rep_home_gpu+' ; rm -r BOS_mean_forcast " ')
	    os.system('scp -r '+rep_AD_results+'/CHARME_BOS'+' '+server_gpu_id+':'+rep_home_gpu+'/BOS_mean_forcast > log_scp_bos')

	#    copy each input density field into the created directory
	    os.system('scp '+rep_AD_results+'/inputs/'+filename+' '+server_gpu_id+':'+rep_home_gpu+'/BOS_mean_forcast/input/Calcul_CFD/ > log_scp_input_vol_bos')

	#    run raytracing code on each member
	#    tranfert BOS outputs files to input folder for analysis step

	    print('	')
	    print("***************** Submit job for raytracing ************************** ")
	    print('	')

	    os.system('ssh -t '+server_gpu_id+' "cd '+rep_home_gpu+'/BOS_mean_forcast/ ; python3.6 change_conf_ini.py mean ;'+rep_code_gpu+' ./CONF.ini > log_raytracer ; scp -r output/Deviations '+server_cpu_id+':'+rootdir+'/Deviation_mean " ')

	    print('	')
	    print('************** Wait for observation on mean forcast state *************** ')
	    print('	')
	    
	    while  True :
	       print('Is dev1x.dat imported :'+ str(os.path.isfile( rootdir+'/Deviation_mean/dev1x.dat' )) )
	       if ( os.path.isfile( rootdir+'/Deviation_mean/dev1x.dat' ) ):
                 break
	       time.sleep(5)
	       continue

	    time.sleep(5)

    DAY = date.today().day
    MONTH = date.today().month
    YEAR = date.today().year
    now = datetime.now().time() # time object
    current_time = now.strftime("%H:%M:%S")
    print('	')
    print('********************************************************************** ')
    print('************** Pre-process state and its observation ***************** ')
    print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
    print('********************************************************************** ')
    print('	')


# Preprocessing of the ensemble of observed state
# walk through each member "Deviation_" directory
    size_obs = 0
    for root, subdirs, files in os.walk(rootdir):
      for subdir in subdirs: 
         if ( subdir[:10] == "Deviation_" ):
           member = str( subdir[len("Deviation_"):] )
           print("Entering "+str(subdir)+" :" )
           os.chdir(str(os.path.join(root, subdir)))
           if ( subdir != "Deviation_obs" and subdir != "Deviation_mean" ):
             if ( os.path.isfile( '../obs_state_'+'{:02d}'.format(int(member))+'.txt' ) ):
               print(" Observation vector already exist " )
               continue
           obs_ = []
           for file in glob.glob(prefix_out_bos_filename+"*"):
            print(" deviations are collected from file : " + str(file) )
            dev = loadImageRaw(os.path.join(root+"/"+subdir, file), [n_row,n_col])
            dev_cam_i = dev.flatten()
            obs_.extend(dev_cam_i)
           size_obs = len(obs_)

           if ( subdir != "Deviation_obs" and subdir != "Deviation_mean" ):
             np.savetxt('../obs_state_'+'{:02d}'.format(int(member))+'.txt', obs_)
           elif ( subdir == "Deviation_mean" ):
             np.savetxt('../obs_state_mean.txt', obs_)
           else :
             np.savetxt('../obs.txt', obs_)

    
    os.chdir(rootdir)
    print('	')
    print("   Observation vector size (for PDAF): "+str(size_obs)) 
    print('	')

 
    columns_to_keep = ["Points:0","Points:1","Points:2","Rho","V:0","V:1","V_MV","T","K","mut"]
    state_columns = ["Rho"]
 
    size_state = 0
    size_uncertain = 0
    if True:
       for m in range(N_ens):
         member = str(m+1)
         if ( os.path.isfile( 'ens_'+'{:02d}'.format(int(member))+'.txt' ) ):
              print(" state vector already exist " )
              if ( m == 0 ):
                for col in range(len(uncertain_columns)):
                  param_filename = rootdir+'/weights_'+uncertain_columns[col]+'_'+'{:02d}'.format(int(member))+'.csv'
                  data = pd.read_csv(param_filename, usecols=[0], names=['X'], header=None)
                  param_data = data['X']
                  NbModes = len(param_data)
                data = pd.read_csv('ens_'+'{:02d}'.format(int(member))+'.txt', usecols=[0], names=['X'], header=None)
                size_state = len(data['X']) 
                size_uncertain = NbModes*len(uncertain_columns)
              continue

         # read CFD state 
         state_ = []

         ##### Get state fields
         filename = rootdir+'/'+prefix_out_cedre_filename+member+'.csv'
         data = pd.read_csv(filename, usecols=columns_to_keep, skiprows= 0, sep=',')
         for col in range(len(state_columns)):
           state_.extend(data[state_columns[col]])

         # Augmenting the state with parameters
         for col in range(len(uncertain_columns)):
           param_filename = rootdir+'/weights_'+uncertain_columns[col]+'_'+'{:02d}'.format(int(member))+'.csv'
           data = pd.read_csv(param_filename, usecols=[0], names=['X'], header=None)
           param_data = data['X']
           NbModes = len(param_data)
           state_.extend(param_data)

         size_uncertain = NbModes*len(uncertain_columns)
         size_state = len(state_)

         # write augmented state for analysis step 
         np.savetxt('ens_'+'{:02d}'.format(int(member))+'.txt', state_)

    print('	')
    print("   State vector size (for PDAF): "+str(size_state))
    print('	')

    size_state_pdaf = size_state

    time.sleep(5)
    
    print('	')
    print("********************* compute MRMSE_OBS ************************** ")
    print('	')
    data = pd.read_csv(rootdir+'/obs.txt', usecols=[0], names=['X'], header=None)
    exp_obs_ = list(data['X'])
    data = pd.read_csv(rootdir+'/obs_state_mean.txt', usecols=[0], names=['X'], header=None)
    mean_f_obs_ = list(data['X'])
    RMSE = np.array(mean_f_obs_) - np.array(exp_obs_)
    RMSE_l = list(RMSE**2)

    MRMSE_OBS = Average(RMSE_l)**0.5
    print(" MRMSE_OBS : "+str(MRMSE_OBS))

    np.savetxt(rep_AD_results+'/inputs/MRMSE_OBS_it_'+str(it-1)+'.txt', [MRMSE_OBS])

#*************************************************************************************************#
#*************************************************************************************************#

    DAY = date.today().day
    MONTH = date.today().month
    YEAR = date.today().year
    now = datetime.now().time() # time object
    current_time = now.strftime("%H:%M:%S")
    print('	')
    print('********************************************************************** ')
    print('******************** Proceed for the analysis step ******************* ')
    print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
    print('********************************************************************** ')
    print('	')

    rep_analysis = rep_AD_results + "/Code_analysis/aeroThDA_parallel/PDAF_offline"
       
    if ( os.path.isfile( rep_AD_results+'/log_pdaf_'+str(N_ens)) and 
         check_if_string_in_file( rep_AD_results+'/log_pdaf_'+str(N_ens), "END") ):
	    print('	')
	    print("******************** Analysis step already done with pdaf ***************************** ")
	    print('	')
    else :

	    print('	')
	    print("******************** Submit job for pdaf ***************************** ")
	    print('	')
	    #os.system('ssh -t '+server_cpu_id_spiro+' "sbatch /stck/mbenali/job_spiro_analysis_prim.sh -e '+rep_analysis+' -o '+rep_AD_results+' -n '+str(size_state_pdaf)+' -m '+str(size_obs)+' -N '+str(N_ens)+' -r '+str(rms_obs)+' -f '+str(forget)+'" ')
	    os.system('cd '+rep_AD_results+' ; mpirun -np 1 '+rep_analysis+' -n_cfd '+str(size_state_pdaf)+' -dim_obs '+str(size_obs)+' -dim_ens '+str(N_ens)+' -filtertype 4 -type_trans 0 -type_sqrt 0 -rms_obs '+str(rms_obs)+' -forget '+str(forget)+' > '+rep_AD_results+'/log_pdaf_'+str(N_ens))

	    while  True :
	       if ( os.path.isfile( rep_AD_results+'/log_pdaf_'+str(N_ens)) and 
		 check_if_string_in_file( rep_AD_results+'/log_pdaf_'+str(N_ens), "END") ):
                 time.sleep(5) 
                 break
	       continue

    DAY = date.today().day
    MONTH = date.today().month
    YEAR = date.today().year
    now = datetime.now().time() # time object
    current_time = now.strftime("%H:%M:%S")
    print('	')
    print("******************** Analysis step done ***************************** ")
    print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
    print('	')

#*************************************************************************************************#
#*************************************************************************************************#

    print('	')
    print('**************** Save new uncertain parameters into cilepi readable input ***************** ')
    print('	')
 
    # Regular meshgrid defintion (Box domain)

    myIndices = [res_x-1, res_y-1]
    myMesher = ot.IntervalMesher(myIndices)
    lowerBound = [0.0, 0.0]
    upperBound = [max_xtp*D, max_ytp*D]
    myInterval = ot.Interval(lowerBound, upperBound)
    meshd = myMesher.build(myInterval)
    vertices = meshd.getVertices()

    for m in range(N_ens):
      member = str(m+1)
      print('     member '+member)
      if ( os.path.isfile( rep_AD_results+'/outputs/execute_cilepi_'+'{:02d}'.format(int(member))+'.log') and check_if_string_in_file( rep_AD_results+'/outputs/execute_cilepi_'+'{:02d}'.format(int(member))+'.log', "ECRITURE DU FICHIER TERMINEE") ):
        continue 
      data = pd.read_csv(rep_AD_results+'/outputs/ens_'+'{:02d}'.format(int(member))+'_ana.txt', usecols=[0], names=['X'], header=None)
      state_a = list(data['X'])
      all_weights =state_a[len(state_a)-size_uncertain:]

    # Define dictionary
      offset = 0
      for col in range(len(uncertain_columns)): 
        weights=[]
        for w in range(NbModes):
          weights.append(all_weights[w+offset])
        offset = NbModes

        # write new parameters in files
        np.savetxt(rep_AD_results+'/outputs/weights_'+uncertain_columns[col]+'_'+'{:02d}'.format(int(member))+'.csv', weights)
             
        ##### write uncertainties in file at output dir. 
        uncertain_df = pd.read_csv(rootdir+'/uncertain_'+'{:02d}'.format(int(member))+'_cilepi.csv', usecols=[i for i in range(len(all_uncertain_columns))], names=all_uncertain_columns, sep=' ')

        # Weighting sum of KL modes 
        uncertain = np.zeros((len(uncertain_df[all_uncertain_columns[0]]),))
        for i in range(NbModes):
          Ev_filename = rootdir+'/phi_'+uncertain_columns[col]+'_'+'{:02d}'.format(i)+'.csv'
          data = pd.read_csv(Ev_filename, usecols=[0], names=['X'], header=None)
          phi_i= data['X']
          uncertain += np.array(weights[i]*phi_i)

        #plotInterpolatedField(uncertain, vertices, res_y, res_x)
        # write into dataframe
        for col2 in range(len(all_uncertain_columns)):
          if ( uncertain_columns[col] != all_uncertain_columns[col2] ):
            continue
          else :
            uncertain_df[all_uncertain_columns[col2]] = uncertain   
        uncertain_df.to_csv(rep_AD_results+'/outputs/uncertain_'+'{:02d}'.format(int(member))+'_cilepi.csv', index=False, header=False, sep=' ')

      os.system('ssh -t '+server_cpu_id+' "cp '+rep_AD_results+'/outputs/uncertain_'+'{:02d}'.format(int(member))+'_cilepi.csv '+rep_AD_code+'/model_forcast_'+member+'/state_source_for_cilepi.csv "')
      os.system('ssh -t '+server_cpu_id+' "cp '+rootdir+'/coords_space_for_cilepi.csv '+rep_AD_code+'/model_forcast_'+member+'/. " ')
      
      os.system('ssh -t '+server_cpu_id+' " source /etc/profile.d/modules.sh ; module use /opt/tools/modules/compilation; module use /opt/tools/modules/librairies; module use /opt/tools/modules/logiciels; module use /opt/tools/modules/mpi; module use /opt/tools/modules/obsoletes; module use /opt/tools/modules/optimisation; module use /opt/tools/modules/socles; module use /tmp_user/sator/cedre/module/chaine_cedre; module unload intel/17.0.4; module unload impi/17; module load python/3.6.1; module load intel/18.0.3; module load impi/18; module load cedre/8.1.0.3/intel18_impi18; cd '+rep_AD_code+'/model_forcast_'+member+'/; ../model_forcast/_CILEPI_/fichier_1/cilepi -epicea > execute_cilepi.log 2>&1 ; mv execute_cilepi.log '+rep_AD_results+'/outputs/execute_cilepi_'+'{:02d}'.format(int(member))+'.log " ')
 
      while  True :
        if ( os.path.isfile( rep_AD_results+'/outputs/execute_cilepi_'+'{:02d}'.format(int(member))+'.log') and check_if_string_in_file( rep_AD_results+'/outputs/execute_cilepi_'+'{:02d}'.format(int(member))+'.log', "ECRITURE DU FICHIER TERMINEE") ):
                 time.sleep(1) 
                 break
        continue
    
    print('     sample mean ')
    data = pd.read_csv(rep_AD_results+'/outputs/state_ana.txt', usecols=[0], names=['X'], header=None)
    state_a = list(data['X'])
    all_weights =state_a[len(state_a)-size_uncertain:]

    # Define dictionary
    offset = 0
    for col in range(len(uncertain_columns)) :
        weights=[]
        for w in range(NbModes):
          weights.append(all_weights[w+offset])
        offset = NbModes

        # write new parameters in files
        np.savetxt(rep_AD_results+'/outputs/weights_'+uncertain_columns[col]+'_mean.csv', weights)
             
        ##### write uncertainties in file at output dir. 
        uncertain_df = pd.read_csv(rootdir+'/uncertain_mean_cilepi.csv', usecols=[i for i in range(len(all_uncertain_columns))], names=all_uncertain_columns, sep=' ')

        # Weighting sum of KL modes 
        uncertain = np.zeros((len(uncertain_df[all_uncertain_columns[0]]),))
        for i in range(NbModes):
          Ev_filename = rootdir+'/phi_'+uncertain_columns[col]+'_'+'{:02d}'.format(i)+'.csv'
          data = pd.read_csv(Ev_filename, usecols=[0], names=['X'], header=None)
          uncertain += np.array(weights[i]*data['X'])

        # write into dataframe
        for col2 in range(len(all_uncertain_columns)):
          if ( uncertain_columns[col] != all_uncertain_columns[col2] ):
            continue
          else :
            uncertain_df[all_uncertain_columns[col2]] = uncertain   
        uncertain_df.to_csv(rep_AD_results+'/outputs/uncertain_mean_cilepi.csv', index=False, header=False, sep=' ')

    os.system('ssh -t '+server_cpu_id+' "cp '+rep_AD_results+'/outputs/uncertain_mean_cilepi.csv '+rep_AD_code+'/model_forcast_mean/state_source_for_cilepi.csv "')
    os.system('ssh -t '+server_cpu_id+' "cp '+rootdir+'/coords_space_for_cilepi.csv '+rep_AD_code+'/model_forcast_mean/. " ')
      
    os.system('ssh -t '+server_cpu_id+' " source /etc/profile.d/modules.sh ; module use /opt/tools/modules/compilation; module use /opt/tools/modules/librairies; module use /opt/tools/modules/logiciels; module use /opt/tools/modules/mpi; module use /opt/tools/modules/obsoletes; module use /opt/tools/modules/optimisation; module use /opt/tools/modules/socles; module use /tmp_user/sator/cedre/module/chaine_cedre; module unload intel/17.0.4; module unload impi/17; module load python/3.6.1; module load intel/18.0.3; module load impi/18; module load cedre/8.1.0.3/intel18_impi18; cd '+rep_AD_code+'/model_forcast_mean; ../model_forcast/_CILEPI_/fichier_1/cilepi -epicea > execute_cilepi.log 2>&1 ; mv execute_cilepi.log '+rep_AD_results+'/outputs/execute_cilepi_mean.log " ')
 
    while  True :
        if ( os.path.isfile( rep_AD_results+'/outputs/execute_cilepi_mean.log') and check_if_string_in_file( rep_AD_results+'/outputs/execute_cilepi_mean.log', "ECRITURE DU FICHIER TERMINEE") ):
                 time.sleep(1) 
                 break
        continue
 
    print(' Archive inputs/outputs') 
    os.system('cd '+rep_AD_results+' ; mv outputs/* '+dir_iter+str(it)+'/outputs/. ; mv inputs/* '+dir_iter+str(it)+'/inputs/. ; mv log_pdaf* '+dir_iter+str(it)+'/outputs/. ')

    os.system('cd '+rep_AD_results+'/'+dir_iter+str(it)+' ; cp -r inputs/Deviation_obs '+rootdir+'/. ; cp outputs/uncertain* '+rootdir+'/. ; cp inputs/phi* '+rootdir+'/. ; cp  outputs/weights* '+rootdir+'/. ; cp inputs/coords_space_for_cilepi.csv '+rootdir+'/.  ')

    print('	')
    print('********************************************************************** ')
    print('************ End of the iteration '+str(it)+'of assimilation process ************ ')
    print('Current Time = '+str(DAY)+'/'+str(MONTH)+'/'+str(YEAR)+', '+current_time)
    print('********************************************************************** ')
    print('	')

    it += 1
    while True :
      now = datetime.now().time()
      current_time = now.strftime("%H:%M:%S")
      hour = int(current_time.split(":")[0])
      print(' hour : '+str(hour))
      minute = int(current_time.split(":")[1])
      if ( hour < 2 or hour > 6 ):
        time.sleep(10)
        break
      time.sleep(600)
      continue

if __name__ == '__main__':
    main()

