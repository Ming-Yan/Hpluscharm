import os
def read_xs(files):
    import json

    f = open(files)
    data = json.load(f)
    xs_dict={}
    for obj in data:
        xs_dict[obj['process_name']]=float(obj['cross_section'])
    return xs_dict
def scale_xs(hist,lumi,events,unscale=False,xsfile="metadata/xsection.json"):

    xs_dict = read_xs(os.getcwd()+"/"+xsfile)
    scales={}

    for key in events:
        if type(key) != str or key=="Data": continue
        if unscale: 
            scales[key]=events[key]/xs_dict[key]*lumi
        else :scales[key]=xs_dict[key]*lumi/events[key]
    hist.scale(scales, axis="dataset")
    return hist
