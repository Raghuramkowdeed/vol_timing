# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:43:26 2018

@author: raghuramkowdeed
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime as dt
import os
import dateutil

from scipy.optimize import linprog

def get_expiry(this_date, fut_data_dict, offset = 5):    
    
    
    y = this_date.year
    m= this_date.month
    key = str(y)+'_'+str(m)
    this_df = fut_data_dict[key]
    expiry = this_df.index[-offset]
    
    #print(expiry)
    if expiry < this_date :
        next_date = this_date + dateutil.relativedelta.relativedelta(months=1)
        #print(next_date)
        y = next_date.year
        m= next_date.month
        key = str(y)+'_'+str(m)
        #print(key)
        this_df = fut_data_dict[key]
        expiry = this_df.index[-offset]
        #print(expiry)
    
    ttm = (expiry - this_date).days
    
    return(expiry,ttm)

def get_fut_curve(this_date, fut_data_dict, vix_index, num_fut= 4):
    steps = range(num_fut)
    p_vec = []
    next_p_vec = []
    expiry, ttm = get_expiry(this_date, fut_data_dict)    
    
    for step in steps:    

        next_expiry = expiry + dateutil.relativedelta.relativedelta(months=step)

        y = next_expiry.year
        m= next_expiry.month
        key = str(y)+'_'+str(m)
        this_df = fut_data_dict[key]
        p = this_df.loc[this_date]
        p_vec.append(p['Close'])
        next_p_vec.append(p['next_Close'])
    
    p_vec = pd.Series(p_vec, index = steps, name = this_date)
    next_p_vec = pd.Series( next_p_vec, index = steps, name = this_date)
    return p_vec, next_p_vec, expiry, ttm

def get_const_fut( this_date, fut_data_dict, vix_index, num_fut ):
    p_vec, next_p_vec, expiry, ttm = get_fut_curve(this_date, fut_data_dict, vix_index, num_fut)
    
    num_days = 31.0
    
    w1 = (ttm)/(num_days)
    w2 = 1.0 - w1
    
    const_p_vec = []
    const_next_p_vec = []
    steps = range(num_fut-1)
    for i in steps:
        p = ( p_vec[i]*w1 ) + (p_vec[i+1]*w2)
        n_p = ( next_p_vec[i]*w1 ) + (next_p_vec[i+1]*w2)
        
        const_p_vec.append(p)
        const_next_p_vec.append(n_p)
        
    const_p_vec = pd.Series(const_p_vec, index = steps, name = this_date)   
    const_next_p_vec = pd.Series(const_next_p_vec, index = steps, name = this_date)   
    
    return const_p_vec, const_next_p_vec, expiry, ttm


def get_carry(this_date, const_p_vec, vix_index, num_fut):
    this_vec = pd.Series(np.zeros(num_fut), name = this_date)
    this_vec.iloc[0] = vix_index.loc[this_date]
    this_vec.iloc[1:] = const_p_vec.values
    carry = this_vec.pct_change()/31.0
    carry = carry.iloc[1:]
    return carry


class VixManager():

      def __init__(self, fut_data_dir, index_file,  ):
          fut_files = os.listdir(fut_data_dir)
          fut_data_dict = {}

          for this_file in fut_files :
              this_df = pd.read_csv(fut_data_dir+'/' + this_file, index_col = 'Unnamed: 0')
              this_df.index = [ dt.datetime.strptime( val ,'%Y-%m-%d') for val in this_df.index ] 
              this_df = this_df[['Close']]
              this_df['next_Close'] = this_df['Close'].shift(-1)
              this_df = this_df.iloc[:-1,:]
    
              fut_data_dict[this_file] = this_df
          self.fut_data_dict = fut_data_dict
          
          vix_index = pd.read_csv(index_file, index_col = 'Date')
          vix_index.index = [ dt.datetime.strptime( val ,'%m/%d/%Y') for val in vix_index.index ]
          vix_index = vix_index['VIX Close']
          self.vix_index = vix_index
          
          self.const_fut_df = pd.DataFrame()
          self.const_fut_df_next = pd.DataFrame()
          self.const_fut_carry_df = pd.DataFrame()
          self.const_fut_cov_df = pd.DataFrame()
          self.const_fut_beta_df = pd.DataFrame()
    
      def set_const_fut_data(self, start_date, end_date, cov_hl = 120, num_fut = 6):
          self.const_fut_df = pd.DataFrame()
          self.const_fut_df_next = pd.DataFrame()
          
          self.const_fut_carry_df = pd.DataFrame()

          self.const_fut_cov_df = pd.DataFrame()
          self.const_fut_beta_df = pd.DataFrame()
          
          self.index_vol = pd.Series()
          
          dates = self.vix_index.index
          dates = dates[(dates>=start_date)&(dates<=end_date)]
          
          for this_date in dates :

              
              p1, p2, e, t = get_const_fut(this_date, self.fut_data_dict, self.vix_index,num_fut)
              
              self.const_fut_df = self.const_fut_df.append(p1)
              self.const_fut_df_next = self.const_fut_df_next.append(p2)

              carry = get_carry(this_date, p1, self.vix_index, num_fut)
              carry = -carry

              self.const_fut_carry_df = self.const_fut_carry_df.append(carry)

        
              
          this_vix_index = self.vix_index.loc[dates]
          
          index_var = (this_vix_index.pct_change()).ewm(halflife=cov_hl).var()
          x1 = self.const_fut_df.pct_change()
          x2 = x1.ewm(halflife=cov_hl)
          x3 = x2.cov(this_vix_index.pct_change())
          const_fut_beta_df = x3 
          
          self.const_fut_beta_df = (const_fut_beta_df.T* ((1/index_var).values) ).T
          self.index_var = index_var
            
          self.const_fut_cov_df = ( (self.const_fut_df.pct_change()).ewm(halflife=cov_hl)).cov()
      
      def get_const_w_ret(self, w, tc = 0.005, margin = 0.5 ):
          
          w = pd.Series(w)
          w = w / ( (w.abs()).sum() )
          
          
          pnl_vec = []
          prev_w = np.zeros(self.const_fut_df.shape[1])
          
          for this_date in self.const_fut_df.index :
              v1 = self.const_fut_df.loc[this_date]
              v2 = self.const_fut_df_next.loc[this_date]
              mc = np.sum(np.abs(w))*margin
              adj_w = w/mc
              
              r = (v2-v1)/v1 
              p = np.dot(r, adj_w)
              
              t_cost = (adj_w-prev_w).abs().sum()*tc
              p = p - t_cost
              prev_w = adj_w
              
              pnl_vec.append( p )
          
          
          pnl_vec = pd.Series(pnl_vec, index = self.const_fut_df.index)
          return pnl_vec

      def get_best_sharpe_w_ret(self,  tc = 0.005, margin = 0.3):
          pnl_vec = []
          dates = []
          prev_w = np.zeros(self.const_fut_df.shape[1])
          
          for i, this_date in enumerate( self.const_fut_df.index) :
              this_cov = self.const_fut_cov_df.iloc[i,:,:]
              print(this_cov.shape) 
              this_cov = np.linalg.inv(this_cov)
              carry = self.const_fut_carry_df.loc[this_date]
              
              w = np.dot(this_cov, carry)      
              w = pd.Series(w)
              w = w / ( (w.abs()).sum() )
              print(w)

              
              mc = np.sum(np.abs(w))*margin
              adj_w = w/mc
              
              if w.isnull().any() :
                  print(this_date)
                  continue
                            
              
              v1 = self.const_fut_df.loc[this_date]
              v2 = self.const_fut_df_next.loc[this_date]              
               
              r = (v2-v1)/v1 
              
      
              p = np.dot(r, adj_w)
              t_cost = ( (adj_w-prev_w).abs().sum()*tc )*margin
              #print(t_cost)
              p = p - t_cost
              prev_w = adj_w

              pnl_vec.append( p )
              dates.append(this_date)
          
          
          pnl_vec = pd.Series(pnl_vec, index = dates)
          return pnl_vec
         
         
      def get_zero_beta_w_ret(self,  tc = 0.005, margin = 0.3):
        
          pnl_vec = []
          dates = []
          prev_w = np.zeros(self.const_fut_df.shape[1])
          
          for i, this_date in enumerate( self.const_fut_df.index) :
              if i <= 30 :
                 continue
              
#              this_cov = self.const_fut_cov_df.iloc[i,:,:]
#              this_cov = np.linalg.inv(this_cov)
              carry = self.const_fut_carry_df.loc[this_date]
              beta = self.const_fut_beta_df.loc[this_date]
              #print(beta.shape)
              #print(carry.shape)

              c = self.const_fut_carry_df.iloc[i,:]
              A = self.const_fut_beta_df.iloc[i:(i+1),:]
              b = [0.0]

              x_bounds = (-1, 1)

              res = linprog(c, A_eq=A, b_eq=b, bounds=(x_bounds),
                    options={"disp": False})

                    
              
              w = -res.x      
              w = pd.Series(w)
              w = w / ( (w.abs()).sum() )
                
            
                


              
              mc = np.sum(np.abs(w))*margin
              adj_w = w/mc
                
              this_df = pd.DataFrame()
              this_df['c'] = self.const_fut_carry_df.iloc[i,:].values
              this_df['b'] =  self.const_fut_beta_df.iloc[i,:].values
              this_df['w'] =  adj_w
              print(this_df)  
              
              if w.isnull().any() :
                  print(this_date)
                  continue
                            
              
              v1 = self.const_fut_df.loc[this_date]
              v2 = self.const_fut_df_next.loc[this_date]              
               
              r = (v2-v1)/v1 
              
      
              p = np.dot(r, adj_w)
              t_cost = ( (adj_w-prev_w).abs().sum()*tc )*margin
              #print(t_cost)
              p = p - t_cost
              prev_w = adj_w

              pnl_vec.append( p )
              dates.append(this_date)
          
          
          pnl_vec = pd.Series(pnl_vec, index = dates)
          return pnl_vec
         

      def get_zero_carry_w_ret(self,  tc = 0.005, margin = 0.3):
        
          pnl_vec = []
          dates = []
          prev_w = np.zeros(self.const_fut_df.shape[1])
          
          for i, this_date in enumerate( self.const_fut_df.index) :
              if i <= 30 :
                 continue
              
#              this_cov = self.const_fut_cov_df.iloc[i,:,:]
#              this_cov = np.linalg.inv(this_cov)
              carry = self.const_fut_carry_df.loc[this_date]
              beta = self.const_fut_beta_df.loc[this_date]
              #print(beta.shape)
              #print(carry.shape)

              c = -self.const_fut_beta_df.iloc[i,:]
              A = self.const_fut_carry_df.iloc[i:(i+1),:]
              b = [0.0]

              x_bounds = (-1, 1)

              res = linprog(c, A_eq=A, b_eq=b, bounds=(x_bounds),
                    options={"disp": False})

                    
              
              w = -res.x      
              w = pd.Series(w)
              w = w / ( (w.abs()).sum() )
              print(w)
              
              mc = np.sum(np.abs(w))*margin
              adj_w = w/mc
                
              this_df = pd.DataFrame()
              this_df['c'] = self.const_fut_carry_df.iloc[i,:].values
              this_df['b'] =  self.const_fut_beta_df.iloc[i,:].values
              this_df['w'] =  adj_w
              print(this_df)   
            
            
              if w.isnull().any() :
                  print(this_date)
                  continue
                            
              
              v1 = self.const_fut_df.loc[this_date]
              v2 = self.const_fut_df_next.loc[this_date]              
               
              r = (v2-v1)/v1 
              
      
              p = np.dot(r, adj_w)
              t_cost = ( (adj_w-prev_w).abs().sum()*tc )*margin
              #print(t_cost)
              p = p - t_cost
              prev_w = adj_w

              pnl_vec.append( p )
              dates.append(this_date)
          
          
          pnl_vec = pd.Series(pnl_vec, index = dates)
          return pnl_vec
        
         
     
          