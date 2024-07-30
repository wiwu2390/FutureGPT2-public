import datetime, random, string
from functools import reduce
import operator

def run_id():
    now = datetime.datetime.now()
    rand = ''.join(random.choices(string.hexdigits, k=5))
    return '-'.join([now.strftime("%Y%m%d-%H%M%S"), rand])

def try_cast(a):
    try:
        return int(a)
    except ValueError:
        pass
    try:
        return float(a)
    except ValueError:
        pass
    return a

def split_config(config):
    '''
    Hacky way to allow listing of param combinations in wandb sweep config (instead of full grid)
    e.g. an entry like {'a,b': (x,y)} should be split into {'a': x, 'b': y}
    Both updates the input wandb.config and returns a dict of split params
    '''
    new_config = reduce(
        operator.ior,
        [
            dict(zip(k.split(','), v.split(',')))
            if ',' in k and k.count(',') == v.count(',')
            else {k: v}
            for k, v in dict(config).items()
        ],
        {}
    )
    new_config = {k: try_cast(v) for k, v in new_config.items()}
    config.update(new_config)
    return new_config

