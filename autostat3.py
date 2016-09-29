#!/usr/bin/python2.7
import warnings
import time
def customwarn(message, category, filename, lineno, file=None, line=None):
        
        pass
warnings.showwarning = customwarn
import sys, os
import pandas as pd
import optparse
import numpy as np
from abc import ABCMeta, abstractmethod
from glob import glob
import json
from math import *
#ML methods
from sklearn import preprocessing
from sklearn.externals import joblib
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import mean_squared_error
file_location = os.path.dirname(os.path.realpath(__file__))
o=None
global myargv
myargv = []
engines = {}
classification_engines = []
regression_engines     = []
loaded = {}

metrics = [
    ['regression_metrics', "Metrics useful for measuring closeness to a numerical target"],
    ['classification_metrics', "Metrics useful for measuring whether a target was classified correctly"],
    ['confusion_matrix', "Classification metric showing which classes were confused with which other classes"],
    ['confusion_matrix_stats', "Detailed classification metrics derived from the confusion matrix"],
    ['sample_data', "Show some sample predictions"],
    ['none', "Show nothing by default"],
    ['basic', "Show basic metrics"]
]

def make_metrics_text(prefix_spaces=0):
    metrics_text = ""
    for i in metrics: 
        metrics_text += (" " * prefix_spaces) + i[0] + ": " + i[1] + "\r\n"
    return metrics_text

class AutostataCommandLine(object):
    """docstring for AutostataCommandLine"""
    def __init__(self, data_list=None):
        
        p = PreOptionParser()
        p.add_option("-e", "--engine", dest="engine", default=None, help="Explicitly select engine.  List available with 'help'.")
        p.remove_option("-h")
        o, remainder = p.parse_args(myargv)
        self.reportgroup = ''
        self.reportopts = ''
        self.parser = ''
        self.filename =''
      
        if o.engine == "help":
            ns="Autostat_"
            usage= "Available Engines:\n"
            for e in engines.keys():
                if e[0:len(ns)]=="Autostat_": e="_".join(e.split("_")[1:])    
                usage+= "   " + e  + "\n"
            sys.stdout.write(usage)
            sys.exit(11)
        if o.engine!=None:
            engines["Autostat_"+o.engine]["init"]()
        else:
            myargv.append("-e")
            myargv.append("SKLearn")
            #engines["Autostat_SKLearn"]["init"]()
        self.custom_init()
        self.collect_options()
        self.add_options()
        if data_list is not None: # when called from function
            self.df = data_list
            
        else: # when called from command line 
            self.read_file()
        self.absorb()
        
    def collect_options(self):
        
        usage = "%prog [options] [file to model or predict, blank for stdin]\n"
        parser = optparse.OptionParser(usage=usage)
        parser.add_option("-p", "--predict", dest="predict", default=False, action="store_true", help="Predict based on input")
        parser.add_option("-z", "--disable_normalize", dest="normalize", default=True, action="store_false", help="Don't normalize categorical values to integers")
        parser.add_option("-Z", "--disable_normalize_numerical", dest="normalize_numerical", default=True, action="store_false", help="Don't normalize numerical values to 0..1")
        parser.add_option("-t", "--target", dest="target", default=0,  help="Set target column, by number or name")
        parser.add_option("-e", "--engine", dest="engine", default=None, help="Explicitly select engine.  List available with 'help'.")
        #parser.add_option("-S", "--split", dest="split", default=False, action="store_true", help="Train on first half of data, test against the other half")
        parser.add_option("-o", "--output", dest="output", default=None, help="Write predictions here. - for stdout.")
        parser.add_option("-O", "--output-mode", dest="outputmode", default="inplace", help="Set format of output data: bare, test, full, inplace")
        parser.add_option("-r", "--rounds", dest="rounds", default=0, type="int", help="(Try to) tell Engine how many rounds to run")
        self.reportgroup = optparse.OptionGroup(parser, "Reporting Options")
        self.reportgroup.add_option("-v", "--verbose", dest="verbose", default=False, action="store_false", help="Describe process")
        self.reportgroup.add_option("-q", "--quiet", dest="quiet", default=False, action="store_false", help="Reduce as much noise as possible")
        self.reportgroup.add_option("", "--report", dest="reportopts", default="", help="What to report, comma separated.  'help' for list.\n")
        
       # self.add_options()
        self.parser = parser
  
    def add_options(self):
       
        self.parser.add_option_group(self.reportgroup)
      
        if hasattr(self, "o"): return
        self.o, remainder = self.parser.parse_args(myargv)
        self.o.filename=""
        self.o.testfilename=""
        if len(remainder)==1:
            if not self.o.predict:
                self.o.filename = remainder[0]
            else:
                self.o.filename = remainder[0]
        if len(remainder)==2:
                self.o.filename = remainder[0]
                self.o.testfilename = remainder[1]
        #if self.o.filename == self.o.testfilename: self.o.split==True
        if 'help' in self.o.reportopts:
            print make_metrics_text();
            sys.exit(10)
        self.o.reportopts = self.o.reportopts.split(",")
        if len(self.o.reportopts) == 1 and self.o.reportopts[0]=='': self.o.reportopts = ['basic']
        else: self.o.reportopts.append("basic")
       # self.filename = parser_object.o.filename
       # self.reportopts = parser_object.o.reportopts
      
    def read_file(self):
        
        #TODO:  XLS/XLSX
        source=None
        
        if len(self.o.filename)>1: 
            source = open(self.o.filename, "rb")
        else:
            source = sys.stdin

        self.df = pd.read_csv(source)
       
    def absorb(self):
        
        
        targetraw = self.o.target
        if type(targetraw)==str and targetraw.isdigit(): targetraw=int(targetraw)
        if type(targetraw)==int:
            self.o.target = self.df.columns[targetraw]
            self.o.targetcol = targetraw
        elif type(targetraw)==str:
            self.o.targetcol = self.df.columns.tolist().index(targetraw)
            self.o.target    = targetraw
        else: 
            print >> sys.stderr, "Couldn't find column.  Case sensitive?"
            sys.exit(100)


class Autostat(AutostataCommandLine):
    __metaclass__ = ABCMeta

    def __init__(self, data_list=None, autorun=True, o=None):
        
        AutostataCommandLine.__init__(self, data_list)
        usage = "%prog [options] [file to model or predict, blank for stdin]\n"
        self.parser = optparse.OptionParser(usage=usage)
        self.pickler = joblib
        self.engineobj = None
        self.enginestash = {}
        self.predictions = None
        self.results = []
        if o!=None: self.o = o
        self.denormalized = None
        self.featuremap = {}
        self.disable_model_pickle = False
        self.model = None
        if self.o.predict: self.load_model()
        self.normalize()
        if not self.o.predict:
            self.set_model()
            self.train()
            self.save_model()
            self.restart()
        else:
            self.hide_truth()
            self.predict()
            self.absorb_predictions()
            self.report()
            self.write_output()
        

    def custom_init(self):
        pass
    
        
    def normalize(self):
        
        df=self.df
        self.originaldf = df.copy()
        for colname in self.df.columns:
            
            col = self.df[colname]
            colType=None
            if self.featuremap.has_key(colname):
                colType=self.featuremap[colname]['colType']
            elif col.dtype != "object": 
                colType = "numerical"
                if len(col.unique())==2: colType = "categorical"
            else: 
                colType = "categorical"
            
            if colType=="numerical":
                if colname==self.o.target: self.o.mode = "regression"
                if not self.o.normalize_numerical: continue
                
                if not self.o.predict:
                    self.featuremap[colname]={ "colType": "numerical", "scaleFactor": col.max() }
                    col/=col.max()
                else:
                    col/=self.featuremap[colname]["scaleFactor"]
                
                    
            else:
                if colname==self.o.target: self.o.mode = "classification"
                if not self.o.normalize: continue
                if not self.o.predict:
                    # build new map
                    nameToNum = {}
                    numToName = {}
                    categories = col.astype('category').cat.categories.tolist()
                    count=0
                    for name in categories:
                        nameToNum[name]=count
                        numToName[count]=name
                        count+=1
                    self.featuremap[colname] = {
                        "colType":   "categorical",
                        "nameToNum": nameToNum,
                        "numToName": numToName
                    }
                    col=col.astype('category').cat.codes
                else:
                    def get_feature(x):
                        
                        ret = x
                        try: return self.featuremap[colname]["nameToNum"][x]
                        except: pass
                        return float('nan')
                    col=col.apply(get_feature)
            self.df[colname]=col
            if len(self.df[self.o.target].unique().tolist())<16: self.o.mode = "classification"
        self.y = np.array(self.df[self.o.target])
        self.X = np.array(self.df.drop(self.o.target,axis=1))

    def set_model(self):
        
        pass
    
    def train(self):
        
        self.model.fit(self.X, self.y)
        pass
    def save_model(self):
        
        autostat_model = {
                              "o": self.o,
                    "enginestash": self.enginestash,
                    "featuremap":  self.featuremap
        }
        try:
            if not self.disable_model_pickle: autostat_model["model"]=self.model
        except: pass
        self.pickler.dump(autostat_model, file_location+"/modeldir/model.pkl")
        print >> sys.stderr, "Model saved.  Run with -p to measure prediction.  Use -o - to emit predicted CSV to standard out."
        
    def load_model(self):
        
        a = self.pickler.load(file_location+"/modeldir/model.pkl")
        try: self.model = a['model']
        except: pass
        o = a['o']
        self.enginestash = a['enginestash']
        self.featuremap = a['featuremap']
        
        # some options are immune to overwrite
        o.predict = self.predict
        o.output = self.o.output
        o.outputmode = self.o.outputmode
       # self.o = o
        o.reportopts = self.o.reportopts
        self.o = o
        if self.o.target not in self.df.columns:
           self.df.insert(self.o.targetcol, self.o.target, 0, False)
           self.o.reportopts = []

    def hide_truth(self):
        
        if 1: #not self.o.sktsr:
            self.df = self.df.drop(self.o.target, axis=1)   

    def predict(self):
        
        #pass
        try:
            self.predictions=self.model.predict(self.df)
        except:
            self.predictions=self.model.predict(np.array(self.df))
    
    def absorb_predictions(self):
        
        self.df = self.originaldf
        if self.predictions!=None: self.df['PREDICTION']=self.predictions
        #self.df['PREDICTION']*=self.featuremap[self.o.target]["scaleFactor"]
        #self.df.insert(self.o.targetcol, self.o.target, self.backup, False)

    def report(self):
        
        test_df = self.df
        def denormalize(x):
            
            colfeatures = self.featuremap[self.o.target]
            if colfeatures["colType"]=="categorical":
                return colfeatures["numToName"][round(x)]
            elif colfeatures["colType"]=="numerical":
                return colfeatures["scaleFactor"]*x
            else: return x
        # you only need to denormalize the predictions -- the original target data
        # comes from originaldf
        if self.o.normalize_numerical: test_df['PREDICTION']=test_df['PREDICTION'].apply(denormalize)
        summary = []
        if self.o.reportopts == ['basic']:
            summary.append("For more details, add the following strings, like:  --report sample_data,confusion_matrix \n" + make_metrics_text(prefix_spaces=3))
  

        if self.o.mode == "classification": #1#0 and self.o.mode=="classification":
            stats = {}
            t = pd.DataFrame()
            if self.featuremap[self.o.target]['colType']=="numerical":
                t['T']=test_df[self.o.target].astype(float).apply(round)
                def minzero(x):
                   
                   if x<0: return 0
                   return x
                t['P']=test_df['PREDICTION'].astype(float)
                if self.df[self.o.target].min()>=0:
                    t['P']=t['P'].apply(minzero)
                t['P']=t['P'].apply(round)
            else:
                t['T']=test_df[self.o.target]
                t['P']=test_df['PREDICTION']
            test_df['CORRECT']= t['T']==t['P']
            stats[False]=stats[True]=0
            for i in test_df['CORRECT']: stats[i]+=1
            stats['predicted'] = stats[True]/(float)(stats[False]+stats[True])
            cm=ConfusionMatrix(t['T'], t['P'])

            if "confusion_matrix" in self.o.reportopts: summary.append(str(cm))
            if "confusion_matrix_stats" in self.o.reportopts: 
                try: summary.append(str(cm.stats_class()))
                except: pass
                try: cm.print_stats() #summary.append(repr(cm.stats_class()))
                except: raise
            if "classification_metrics" in self.o.reportopts:
                summary.append(str(classification_report(t['T'], t['P'], target_names=self.o.target)))
            if "basic" in self.o.reportopts:
                summary.append("Prediction Level: %2.2f (%s false, %s true)" % (stats["predicted"], stats[False], stats[True]))
              
        if 1: #self.o.mode == "regression": #:# or self.o.mode=="regression":
            try: 
                self.outcols = [self.o.target, 'PREDICTION']
                if 'DIFFERENCE' in self.df.columns: self.outcols.append('DIFFERENCE')
                if 'CORRECT' in self.df.columns: self.outcols.append('CORRECT')
                #if "sample_data" in self.o.reportfields: summary.insert(0,self.df[self.outcols])
                if "sample_data" in self.o.reportopts: summary.insert(0,self.df[self.outcols])
                rmse = sqrt(mean_squared_error(test_df[self.o.target], test_df['PREDICTION']))
                test_df['DIFFERENCE'] = abs(test_df[self.o.target]-test_df['PREDICTION'])
                #if "basic" in self.o.reportfields: summary.append("RMSE: %.3f, Diff Average: %.3f, Diff Stddev: %.3f, Target Min: %.3f, Target Max: %.3f" % (rmse, test_df['DIFFERENCE'].mean(), test_df['DIFFERENCE'].std(), test_df[self.o.target].min(), test_df[self.o.target].max()))
                if "self_score" in self.o.reportopts: summary.append("Self Score: " + str(self.model.score()))
                if "regression_metrics" in self.o.reportopts: summary.append("RMSE: %.3f, Explained Variance: %.3f, R2 Score: %.3f, Diff Average: %.3f, Diff Stddev: %.3f, Target Min: %.3f, Target Max: %.3f" % \
                   (rmse, explained_variance_score(t['T'], t['P']), r2_score(t['T'], t['P']), test_df['DIFFERENCE'].mean(), test_df['DIFFERENCE'].std(), test_df[self.o.target].min(), test_df[self.o.target].max()))
                if "basic" in self.o.reportopts: summary.append("Average Error: %.3f   (%s ranges from %.3f to %.3f)" % (test_df['DIFFERENCE'].mean(), self.o.target, test_df[self.o.target].min(), test_df[self.o.target].max()))
                #if "basic" in self.o.reportopts: summary.append("RMSE: %.3f, Diff Average: %.3f, Diff Stddev: %.3f, Target Min: %.3f, Target Max: %.3f" % (rmse, test_df['DIFFERENCE'].mean(), test_df['DIFFERENCE'].std(), test_df[self.o.target].min(), test_df[self.o.target].max()))
            except: pass
        

        for s in summary: print >> sys.stderr, s
        self.results = summary
        
    
    def write_output(self):
        
        df = self.df
        out_df = ""
        
        if self.o.outputmode=="bare": out_df=df['PREDICTION']
        if self.o.outputmode=="test":
            out_df=df[self.outcols]
        if self.o.outputmode=="full": out_df=df
        if self.o.outputmode=="inplace":
            out_df=df.copy()
            out_df[self.o.target] = out_df['PREDICTION']
            try: out_df=out_df.drop('PREDICTION', axis=1)
            except: pass
            try: out_df=out_df.drop('DIFFERENCE', axis=1)
            except: pass
            try: out_df=out_df.drop('CORRECT', axis=1)
            except: pass
        if type(out_df)==str:
            print >> sys.stderr, "Invalid Mode"
            sys.exit(7)
        if self.o.output == "-": self.o.output = sys.stdout
        try: out_df.to_csv(self.o.output, index=False)
        except IOError: pass
        except: raise
        
    def restart(self):
        
        if self.o.testfilename and not self.o.predict:
            self.o.predict = True
            self.o.filename = self.o.testfilename
            engines["Autostat_"+self.o.engine]["init"](o=self.o)

            pass

class PreOptionParser(optparse.OptionParser):
    def exit(self, str):
        
        pass
    def error(self, str):
        
        pass


def register_engine(init, canHandle={}):
    
    name = str(init.__name__)
    
    engine=engines[name]={}
   
    engine["init"] = init
    
    engine["canHandle"] = {}
    

def jank_read(plugin):
    
    if loaded.has_key(plugin): return ""
    loaded[plugin]=True
    j = open(plugin, "r").read()
    return str(j)


def common_code(myargv, data_list=None):
    print "JAWAD FAISAL"
    try: 
        exec(jank_read(file_location+"/autostat_plugin_sk_all.py"),globals()) # others may take a dependency on this
    except: pass

    for plugin in glob(file_location+"/autostat_plugin_*.py"):

        j = jank_read(plugin)
        try:
            pass
            exec(j,globals())
        except: pass
    print "2ND JAWAD FAISAL"    
    AutostataCommandLine()
    engines["Autostat_SKLearn"]["init"]()
    print "LAST"
   
    
myargv = sys.argv
myargv.pop(0)
if len(myargv)>0:
    common_code(myargv)
   
def autostat(data , args, output):
    
    
    option_list = {
    'predict':'--predict',
    'disable_normalize':'--disable_normalize',
    'split':'--split',
    'target':'--target',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5'
    }
    for item in args:
        myargv.append(option_list[item])

    if type(output) is list and len(output)>0:

        if len(output)==2:
            file_name  = output[0]
            output_mode = output[1]
        elif len(output)==1:
            file_name  = '-'
            output_mode = output[0]
        
        myargv.append('-o')
        myargv.append(file_name)        
        myargv.append('-O')
        myargv.append(output_mode)

    if type(data) is list:
        data_list = pd.DataFrame(data, columns=['TARGET','ID','COMPANY','SALARY','JOBTIME'])
    else:
        data_list = None
        myargv.append(data)
    
   
        
    res = common_code(myargv, data_list)
     
    return res
    
