# -*- -*- -*- -*- -*- -*- -*- -*- 
"""
Prepared using : Python version 3.8.13 | Spyder IDE version 5.1.5
Author : Amarnath Mitra | Email : amarnath.mitra@gmail.com
Updated : Jul, 2022
"""
# -*- -*- -*- -*- -*- -*- -*- -*- 

# ===============================================
# *** Big Data & Data Analytics for Managers ***
# ===============================================

# --------------
# --- Topics ---
# --------------

# 2. Data Manipulation using Numpy & Pandas
# 2.1. Arrays and Matrices
# 2.2. Dataframes : Create, Read/Write, Subset
# 2.3. Reshape Dataframes : Melt, Pivot, Group
# 2.4. Combine Dataframe : Merge, Append
# 2.5. Describe Data


# *************************************************************************************************

# -------------------------------------------
# --- Python Built-in Modules & Libraries ---
# -------------------------------------------

'''
import xxx as x  # xxx : Python Built-in Module | x : Alias (short-name for quick & easy reference)

from xxx import zzz as z  # xxx : Main Module | zzz : Sub Module or Function
'''

# Important Python Built-in Moduldes
'''
1. array, numbers, random, math, statistics	: Numeric Data Types & Math-Stat related modules
2. calendar, datetime, time, timeit : Date & Time related modules
3. type, copy 				
4. csv, json, xml, sqlite3, tarfile, zipfile : File Format & Language related modules
5. html, http, webbrowser, email : Internet Interface related modules		
6. functools, trace, logging, warnings : Programing or Coding related modules		
7. sched, multiprocessing, threading : Scheduling, Multiprocessing & Multithreading related modules			
8. os, sys, shutil : OS, System & File Handling related modules			
9. string, re : Text & Regular Expression related modules				
10. tkinter : GUI related module
'''

# Other Important Python Libraries	
'''
numpy : Array related library
pandas : Dataframe related library
matplotlib : Plotting | Visualization library
statsmodels : Statistical Analysis library
scikit-learn : Machine Learning library
'''

				
# *************************************************************************************************


# =================================================
# --- 2. Data Manipulation using Numpy & Pandas ---
# =================================================

# 2.1. Numpy : Arrays & Matrices
# ------------------------------

'''
1. Numpy Library is used to Create and Manipulate n-Dimensional Arrays & Matrices.
2. Manipulation of Numpy Arrays & Matrices are Better & Faster than Native Python Arrays & Matrices and hence are Preferred.
3. Application : Linear Algebra 
'''

import numpy as np

# Arrays are Column-wise Vectors
a1 = np.array([1, 2, 3, 4], dtype=float); a1 # 1-d Array 
type(a1)
a2 = np.array([[11, 12, 13, 14], [21, 22, 23, 24]]); a2 # 2-d Array
a3 = np.array([[[11, 12, 13], [21, 22, 23]], [[31, 32, 33], [41, 42, 44]]]); a3 # 3-d Array


# Array Manipulation
a2[0] # 2-d Array Indexing
a2.shape # 2-d Array Shape >> {2 Rows, 3 Columns}
a2.reshape(4,2) # 2-d Array Reshape >> {4 Rows, 2 Columns}
a3[0] # 3-d Array Indexing
a3.shape # 3-d Array Shape >> {2 Layers, 2 Rows, 3 Columns}
a3.reshape(2,3,2) # 3-d Array Reshape >> {2 Layers, 3 Rows, 2 Columns}
a1.reshape(2,2) # 1-d --> 2-d Array
a2.reshape(2,2,2) # 2-d --> 3-d Array
a3.reshape(3,4) # 3-d --> 2-d Array
a2.T # 2-d Transpose : (2, 4) --> (4, 2)
a3.T # 3-d Transpose : (2, 2, 3) --> (3, 2, 2)
a11 = np.array([[1, 2], [3, 4]]); a12 = np.array([[5, 6]])
a11_12 = np.concatenate((a11, a12), axis=None); a11_12 # All Element Concatenation to 1-d Array
a11_12_row = np.concatenate((a11, a12), axis=0); a11_12_row # Row-wise Concatenation
a11_12_column = np.concatenate((a11.T, a12.T), axis=1); a11_12_column # Column-wise Concatenation


# Array Stat
a4 = np.array([1, 23, 4]); a5 = np.array([56, 7, 8])
a45 = np.concatenate((a4, a5)).reshape(2,3); a45
np.amin(a45) # Array Min
np.amin(a45, axis=0) # Column Min over Row
np.amin(a45, axis=1) # Row Min over Column
np.amax(a45) # Array Max
np.amax(a45, axis=0) # Column Max over Row
np.amax(a45, axis=1) # Row max over Column
np.sum(a45) # Array Sum
np.sum(a45, axis=0) # Column Sum over Row
np.sum(a45, axis=1) # Row Sum over Column
np.mean(a45) # Array Mean
np.mean(a45, axis=0) # Column Mean over Row
np.mean(a45, axis=1) # Row Mean over Column
np.std(a45) # Array Std.Dev.
np.std(a45, axis=0) # Column Std.Dev. over Row
np.std(a45, axis=1) # Row Std.Dev. over Column
np.cov(a4, a5) # Covariance
np.corrcoef(a4, a5) # Pearson Correlation


# Array Linear Algebra
a6 = np.array([[1, 3], [2, 4]]); a7 = np.array([[5, 7], [6, 8]])
np.vdot(a6, a7) # Vector Multiplication : {1*5 + 3*7 + 2*6 + 4*8}
np.dot(a6, a7) # Matrix Multiplication : {1*5 + 3*6, 1*7 + 3*8, 2*5 + 4*6, 2*7 + 4*8}
np.matmul(a6, a7) # Matrix Multiplication : {1*5 + 3*6, 1*7 + 3*8, 2*5 + 4*6, 2*7 + 4*8}


from numpy import linalg as npla

a67 = np.dot(a6, a7)
npla.det(a67) # Determinant of Matrix
npla.inv(a67) # Inverse of Matrix
eig_val, eig_vec = npla.eig(a67) # Eigen Value & Eigen Vector of Matix
eig_val, eig_vec
a8 = np.array([[1, 2], [2, 3]]); a9 = np.array([4, 7]) # Eq1 : x + 2y = 4 | Eq2 : 2x + 3y = 7
npla.solve(a8, a9) # Solve System of Linear Equations >> x = 2 | y = 1               


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


# Working Directory
'''
Before working with Python Files:
1. Check the Current Working Directory. It is the Default Working Directory (== Documents).
2. Create a Folder at a Desired Location.
3. Keep all relevant Files in the Desired Folder for easy access.
4. Note the Path of the Created Folder.
5. Change the Current Working Directory to the Desired Folder.
'''

import os
os.getcwd() # Current Working Directory
wd_path = r'C:\Users\***\Desktop\***' # New Working Directory Path : use raw (r) string
os.chdir(wd_path) # Change Working Directory to Desired Location


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


# 2.2. Pandas : Dataframe & Series
# --------------------------------

import pandas as pd

# Create Dummy Dataframe : using Python Dict
my_data = {
    'Name': ['Alexa', 'Bixby', 'Cortana', 'Jarvis', 'Lyra', 'Robin', 'Siri'], 
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'], 
    'Education': ['Postgraduate', 'Graduate', 'Postgraduate', 'Doctorate', 'Graduate', 'Graduate', 'Postgraduate'], 
    'Age': [24, 21, 25, 28, 23, 22, 24], 
    'Monthly_Salary': [27000, 25000, 30000, 35000, 26000, 24000, 28000]
    }
my_df = pd.DataFrame(my_data); my_df

# Write Dataframe
my_df.to_csv('my_csv_file.csv') # Write to CSV Format File with Default Index
my_df.to_excel('my_excel_file.xlsx', index=False) # Write to EXCEL Format File without Index

# Read Dataframe
my_df_csv = pd.read_csv('my_csv_file.csv') # Read CSV File from Current Working Directory
my_df_excel = pd.read_excel('my_excel_file.xlsx') # Read EXCEL File from Current Working Directory

# Pandas Series (with Index)
my_series1 = ['a', 'b', 'c', 'd'] # Natural Index : {0, 1, 2, ...}
ps1 = pd.Series(my_series1); ps1
my_series2 = {'a': 12, 'b': 34, 'c': 56} # Given Index : {'a', 'b', 'c', ...}
ps2 = pd.Series(my_series2); ps2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


# 2.3. Reshape Dataframe : Sort, Subset, Filter, Group, Melt, Pivot
# -----------------------------------------------------------------

my_df
my_df.columns
my_df.index

# Sort Dataframe
my_df.sort_values('Age')  # Sort Values by Specific Column in Ascending Order (by Default)
my_df.sort_values(['Age', 'Monthly_Salary'], ascending=[True, False]) # Sort Values by 2 or more Columns in Ascending or Descending Order

# Subset Dataframe
my_df['Name'] # Extract Data of Specific Column
my_df[['Name', 'Age']] # Extract Data of Multiple Columns
my_df.iloc[1] # iloc[Row_Num, Col_Num] : Extract Data by Specific {Row or Column or Both} Number
my_df.loc[0, 'Name'] # loc[row_name, col_name] : Extract Data by Specific {Row or Column or Both} Name

# Filter Dataframe
my_df[my_df['Gender'] == 'Female'] # Filter Data with Single Criteria
my_df[(my_df['Age'] < 25) & (my_df['Education'] == 'Graduate')] # Filter Data with Multiple Criteria 

# Group Dataframe
my_df.groupby('Education')['Monthly_Salary'].mean()

# Pivot Table from Dataframe 
my_df_pivot = pd.pivot_table(my_df, index='Gender', columns='Education', values='Age', aggfunc='median'); my_df_pivot

# Create Dummy Dataframe : using Python Dict
panel_data = {
    'Company': ['ABC', 'ABC', 'DEF', 'DEF', 'GHI', 'GHI'],
    'Year': [2001, 2002, 2001, 2002, 2001, 2002],
    'Revenue': [1234, 2345, 3456, 4567, 5678, 6789]
    }
panel_df = pd.DataFrame(panel_data); panel_df

# Pivot Dataframe
panel_df_pivot = pd.pivot(panel_df, index='Company', columns='Year', values='Revenue'); panel_df_pivot
panel_df_pivot.reset_index(level='Company', inplace=True); panel_df_pivot

# Melt Dataframe
panel_df_pivot_melt =  pd.melt(panel_df_pivot, id_vars=['Company'], value_name='Revenue'); panel_df_pivot_melt


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


# 2.4. Combine Dataframe : Concat, Merge
# --------------------------------------
my_df_f = my_df[my_df['Gender'] == 'Female']; my_df_f
my_df_m = my_df[my_df['Gender'] != 'Female']; my_df_m

# Concatenate Dataframe [axis=0, axis=1, join={outer, inner}]
my_df_fm = pd.concat([my_df_f, my_df_m], ignore_index=True); my_df_fm

# Merge Dataframe {left, right, inner, outer, cross}
my_df_merged_left = pd.merge(my_df_f[['Education', 'Age']], my_df_m[['Education', 'Monthly_Salary']], on='Education', how='left'); my_df_merged_left
my_df_merged_inner = pd.merge(my_df_f[['Education', 'Age']], my_df_m[['Education', 'Monthly_Salary']], on='Education', how='inner'); my_df_merged_inner


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


# 2.5. Dataframe : Describe Data
# ------------------------------

my_df # Dataframe
my_df.head(n=5) # Displays 1st n Rows 
my_df.tail(n=5) # Displays last n Rows 
my_df.shape # Displays Dataframe Dimension: Number of Rows & Columns
my_df.info() # Displays Index, Column-wise Datatypes & Memory information
my_df.describe() # Displays Summary Statistics for Numerical Columns or Variables (only)


# *************************************************************************************************

