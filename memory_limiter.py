import os
import psutil

#The units for this function are very strange
#see ~/Documents/programming/py_useful/resource_limiter/ for details
#7000*1e6 seems to be a good limit
def limit_memory_as(soft_lim):
  '''Sets the (soft) limit to the process's virtual memory usage'''
  proc = psutil.Process(os.getpid())
  _, hard_lim = proc.rlimit(psutil.RLIMIT_AS)
  proc.rlimit(psutil.RLIMIT_AS, (soft_lim, hard_lim))
  return
