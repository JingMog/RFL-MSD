import os
import time

class LoaderTracer():
    #code by jxwang of pydlp team
    def __init__(self,display_freq,drop_history,logger=None):
        self.time_loader_io = 0
        self.time_loader_others = 0
        self.io_count = 0
        self.others_count =0
        self.last_time_stmap_after_io = None
        self.display_freq = display_freq
        self.drop_history = drop_history
        self.logger = logger

    def update_io_time(self,io_time_start):
        self.last_time_stmap_after_io = time.time()
        self.time_loader_io +=self.last_time_stmap_after_io - io_time_start
        self.io_count+=1
        if (self.io_count) %self.display_freq == 0:
            if self.logger is None:
                print("io    time per batch(%d): %.5f"%(os.getpid(),self.time_loader_io /self.io_count ))
            else:
                self.logger.info("io    time per batch(%d): %.5f"%(os.getpid(),self.time_loader_io /self.io_count ))
            if self.drop_history:
                self.time_loader_io=0
                self.io_count =0 
    
    def update_other_time(self,time_end_others):
        if self.last_time_stmap_after_io is not None:
            self.others_count+=1
            self.time_loader_others += time_end_others - self.last_time_stmap_after_io
            if (self.others_count) %self.display_freq == 0 :
                if self.logger is None:
                    print("outer time per batch(%d): %.5f"%(os.getpid(),self.time_loader_others /self.others_count))
                else:
                    self.logger.info("outer time per batch(%d): %.5f"%(os.getpid(),self.time_loader_others /self.others_count))
                if self.drop_history:
                    self.time_loader_others =0 
                    self.others_count =0 

class LoaderProfiler():
    #code by jxwang of pydlp team
    def __init__(self,loader,diplay_freq =100,drop_history=False, logger=None):
        self.profile      = LoaderTracer(diplay_freq,drop_history, logger=logger)
        self.loader_iter  = iter(loader)
        self.index        = 0
            
    def __iter__(self):
        return self
        
    def __next__(self):        
        time_start_io = time.time()  
        self.profile.update_other_time(time_start_io)        
        # item = self.loader_iter.next()
        item = self.loader_iter._next_data() # 这里使用next会报错,改成_next_data
        self.profile.update_io_time(time_start_io)
        index_past = self.index
        self.index += 1
        return index_past,item    
    # only for debug and test,so weird named
    def _test_debug_info__(self):
        return self.profile
