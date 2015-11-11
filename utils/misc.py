import numpy as np
#####################################################################
#           Pretty printing / figure stuff                          #
#####################################################################

from itertools import cycle

styles = [{'color':'r', 'marker':'o'},
          {'color':'g', 'marker':','},
          {'color':'b', 'marker':'.'},
          {'color':'c', 'marker':'v'},
          {'color':'m', 'marker':'^'},
          {'color':'y', 'marker':'<'},
          {'color':'k', 'marker':'>'},
          {'color':'.5','marker':'s'},]
styler = cycle(styles)

np.set_printoptions(precision=3, linewidth=120, suppress=True)

from collections import defaultdict as defdict

def average_by(*params):
    '''average_by works as follows:
      >>>average_by([2,3,4],[0,1,1])
      [[2.0, 0], [3.5, 1]]
      >>>average_by([2,3,4],[0,1,1],[0,1,1])
      [[2.0, 0, 0], [3.5, 1, 1]]

      i.e. it takes the unique values of all later params and uses them
      to average values of the first parameter.'''
    items = defdict(list)
    tuples = zip(*params)
    for t in tuples:
        items[t[1:]] += [t[0]]
    keys = items.keys()
    keys.sort()

    ans = []
    for k in keys:
        v = items[k]

        v = reduce(add,v)/len(v)
        if isscalar(k):
            ans.append( [v,k] )
        else:
            temp = [v]
            temp.extend(copy(k).tolist())
            ans.append(temp)
    return ans
