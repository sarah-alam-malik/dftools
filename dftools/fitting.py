import pysge
import copy
import numpy as np
import pandas as pd
import scipy.stats
import numdifftools
import iminuit

from cachetools.keys import hashkey

from .stats import poisson_interval
from .draw import process_colours, process_names

def cache(method):
    def cachedmethod(self, *args, **kwargs):
        extra = self.parameters[args[-1]]
        key = hashkey(*([method.__name__]+list(args)+[extra]))
        if key not in self.cache:
            result = method(self, *args, **kwargs)
            self.cache[key] = result
        return self.cache[key]
    return cachedmethod

class NLLModel(object):
    def __init__(
        self, dfdata, dfmc, config, same_bin_widths=False,
        shape_type="shape", smooth_region=1.,
        observable_function = "bin_min, bin_max: (bin_min + bin_max)/2."
    ):
        """
        Initialise the NLLModel class

        Parameters
        ----------
        dfdata : pd.DataFrame
            Histograms for data. Multindex dataframe with the index (region,
            bin_min, bin_max) and columns (sum_w,).

        dfmc : pd.DataFrame
            Histograms for MC, with variational templates. Multiindex dataframe
            with the index (region, process, variation, bin_min, bin_max) and
            columns (sum_w, sum_ww). Ensure bin values match between data and
            MC and each variational template.


        config : dict
            Dictionary to change the config of the NLLModel. Key 'regions' is
            a dict of region names to a list of processes. Key 'parameters' is
            a list of dicts with keys for the parameter [name]s, [value]s,
            [limit]s (tuple of min and max), if it's [fixed] (boolean) and
            the [constraint] type (free, gaussian or gamma). Key
            'scale_functions' is a dict of labels to a lambda function as a
            string with two arguments: the observable x and a dict of parameter
            values. MC stat gamma nuisance parameters should be given the name
            '{region}_mcstat_bin{idx}'

        same_bin_widths : bool (default = False)
            When normalising histograms set the bin width to the identical
            value of 1.

        shape_type : str (default = 'shape')
            Choice of 'shape' or 'shapeN' to control the vertical template
            morphing scheme. 'shape' extrapolates linearly and interpolates
            with a quadratic polynomial (in the nuisances**2) on the
            normalised histograms. 'shapeN' does the same but in the log-space,
            i.e. normalised histogram yields are logged then summed and finally
            exponentiated. This prevents negative values smoothly tending to
            zero instead of tapering.

        smooth_region : float (default = 1.)
            Size of the smooth region for shape parameters. Typically taken
            between -1 and +1 where the quadratic interpolation exists and
            a linear interpolation beyound this.

        observable_function : str (default = 'bin_min, bin_max: (bin_min + bin_max)/2.')
            Lambda string to calculate the observable. Default is the bin
            center.
        """
        self.regions = list(config["regions"].keys())
        self.processes = config["regions"]
        self.shape_morph_nuisances = [
            params["name"]
            for params in config["parameters"]
            if params["constraint"] == "gaussian"
        ]
        self.shape_norm_nuisances = [
            params["name"]
            for params in config["parameters"]
            if params["constraint"] == "gaussian"
        ]
        self.parameters = {
                params["name"]: params["value"]
            for params in config["parameters"]
        }
        self.scale_functions = config["scale_functions"]

        self.shape_type = shape_type
        self.smooth_region = smooth_region

        # Region dependent binning
        self.bins = {}
        self.observables = {}
        self.dbins = {}
        for r in self.regions:
            tdf = dfdata.loc[(r,),:]
            bin_min = tdf.index.get_level_values("bin_min").values
            bin_max = tdf.index.get_level_values("bin_max").values
            self.bins[r] = (bin_min, bin_max)
            self.observables[r] = eval('lambda {}'.format(observable_function))(bin_min, bin_max)
            self.dbins[r] = bin_max - bin_min
            if same_bin_widths:
                self.dbins[r] = np.ones_like(bin_min)

        #self.bins = bins
        #self.dbins = bins[1] - bins[0]
        #self.observable = eval('lambda {}'.format(observable_function))(bins[0], bins[1])
        #if not same_bin_widths:
        #    self.dbins = np.ones_like(bins[0])

        self.dfmc = dfmc
        self.dfdata = dfdata

        self.param_names = []
        self.param_inits = []
        self.param_fixes = []
        self.param_limit = []
        for params in config["parameters"]:
            self.param_names.append(params["name"])
            self.param_inits.append(params["value"])
            self.param_fixes.append(params["fixed"])
            self.param_limit.append(params["limit"])

        # Precomputations for lognorms
        self.mc_nom_sum = {}
        self.mc_up_sum = {}
        self.mc_do_sum = {}
        for r in self.regions:
            for p in self.processes[r]:
                try:
                    self.mc_nom_sum[(r,p)] = (
                        self.dfmc.loc[(r, p, ""), "sum_w"].sum()
                    )
                except KeyError:
                    pass

                for n in self.shape_norm_nuisances:
                    try:
                        self.mc_up_sum[(r, p, n)] = (
                            self.dfmc.loc[(r, p, n+"Up"), "sum_w"].sum()
                        )
                        self.mc_do_sum[(r, p, n)] = (
                            self.dfmc.loc[(r, p, n+"Down"), "sum_w"].sum()
                        )
                    except KeyError:
                        pass

        # Precomputions for morphing
        self.delta_sum = {}
        self.delta_diff = {}

        for r in self.regions:
            for p in self.processes[r]:
                for n in self.shape_morph_nuisances:
                    try:
                        nom = self.dfmc.loc[(r, p, ""), "sum_w"].values
                        up = self.dfmc.loc[(r, p, n+"Up"), "sum_w"].values
                        do = self.dfmc.loc[(r, p, n+"Down"), "sum_w"].values
                    except KeyError:
                        continue

                    nom_norm = nom/(nom*self.dbins[r]).sum()
                    up_norm = up/(up*self.dbins[r]).sum()
                    do_norm = do/(do*self.dbins[r]).sum()

                    if self.shape_type == "shape":
                        up_delta = up_norm - nom_norm
                        do_delta = do_norm - nom_norm
                    elif self.shape_type == "shapeN":
                        up_delta = np.where((up_norm>0)&(nom_norm>0), np.log(up_norm/nom_norm), np.zeros_like(nom_norm))
                        do_delta = np.where((do_norm>0)&(nom_norm>0), np.log(do_norm/nom_norm), np.zeros_like(nom_norm))
                    delta_diff = up_delta - do_delta
                    delta_sum  = up_delta + do_delta

                    self.delta_sum[(r, p, n)] = delta_sum
                    self.delta_diff[(r, p, n)] = delta_diff

        # Precomputation of data
        self.data = {
            r: self.dfdata.loc[(r,),"sum_w"].values
            for r in self.regions
        }

        # Precomputation of mc
        self.mc_proc_sumw = {
            r: self.dfmc.loc[(r, pd.IndexSlice[:], ""), "sum_w"].groupby(["bin_min", "bin_max"]).sum().values
            for r in self.regions
        }

        self.mc_proc_sumww = {
            r: self.dfmc.loc[(r, pd.IndexSlice[:], ""), "sum_ww"].groupby(["bin_min", "bin_max"]).sum().values
            for r in self.regions
        }
        self.mc_proc_neff = {
            r: self.mc_proc_sumw[r]**2/self.mc_proc_sumww[r]
            for r in self.regions
        }

        self.mc_sumw = {
            (r, p): self.dfmc.loc[(r, p, ""), "sum_w"].values
            for r in self.regions
            for p in self.processes[r]
        }
        self.mc_sumww = {
            (r, p): self.dfmc.loc[(r, p, ""), "sum_ww"].values
            for r in self.regions
            for p in self.processes[r]
        }

        # Set asimov dataset and toys
        self.asimov = False
        self.seed = 123456
        self.toy = -1


        # function result cache
        self.cache = {}

    def shape_morph(self, region, process, nuisance):
        theta = self.parameters[nuisance]

        # smooth_step_function = 1/8 * (theta * (3*theta - 10) + 15)
        nuisance_val_norm = theta/self.smooth_region
        nuisance_val_norm2 = nuisance_val_norm**2
        if abs(nuisance_val_norm)>=1.:
            smooth_step = np.sign(nuisance_val_norm)
        else:
            smooth_step = 0.125*nuisance_val_norm*(
                nuisance_val_norm2*(3.*nuisance_val_norm2-10.)+15.
            )

        # theta * (diff + sum*smooth_step_function)
        delta_diff = self.delta_diff[(region, process, nuisance)]
        delta_sum = self.delta_sum[(region, process, nuisance)]
        return 0.5*theta*(delta_diff + delta_sum*smooth_step)

    def shape_morphs(self, region, process):
        _mc_sumw = self.mc_sumw[(region, process)]
        shape_vals_norm = _mc_sumw/(_mc_sumw*self.dbins[region]).sum()
        if self.shape_type == "shapeN":
            shape_vals_norm = np.where(shape_vals_norm>0, np.log(shape_vals_norm), -999)

        for nuisance in self.shape_morph_nuisances:
            if (region, process, nuisance) in self.delta_diff:
                shape_vals_norm += self.shape_morph(region, process, nuisance)

        if self.shape_type == "shapeN":
            shape_vals_norm = np.exp(shape_vals_norm)
        shape_vals_norm = np.maximum(1e-10, shape_vals_norm)

        #_mc_sumw = self.mc_sumw[(region, process)]
        # Avoid possible division by zero which we multiply by again
        return shape_vals_norm*(_mc_sumw*self.dbins[region]).sum() #/_mc_sumw

    def shape_norm(self, region, process, nuisance):
        theta = self.parameters[nuisance]

        # # Take lognorm value from input (should be fixed) parameter with both
        # # up and down variations
        # key = "_".join([nuisance, region, process, "lognorm"])
        # if key+"Up" in self.parameters:
        #     return (
        #         self.parameters[key+"Up"] if theta>=0. else
        #         self.parameters[key+"Down"]
        #     )**(theta)

        nom_sum = self.mc_nom_sum[(region, process)]
        up_sum = self.mc_up_sum[(region, process, nuisance)]
        do_sum = self.mc_do_sum[(region, process, nuisance)]
        return (up_sum/nom_sum if theta>=0. else nom_sum/do_sum)**(theta)

    def shape_norms(self, region, process):
        return np.prod([
            self.shape_norm(region, process, n)
            for n in self.shape_norm_nuisances
            if (region, process, n) in self.mc_up_sum
        ])

    def shapes(self, region, process):
        return self.shape_morphs(region, process)*self.shape_norms(region, process)

    def prediction(self, region, process):
        _nominal_abs = self.mc_sumw[(region, process)]

        if (region, process) in self.scale_functions:
            _nominal_scale = eval('lambda '+self.scale_functions[(region, process)])(
                self.observables[region], _nominal_abs, self.parameters,
            )
        else:
            _nominal_scale = 1.
        _syst_scale = self.shapes(region, process)

        _sumw = self.mc_proc_sumw[region]
        _sumww = self.mc_proc_sumww[region]
        _mcstat_theta = np.array([
            self.parameters["_".join([region, "mcstat", "bin{}".format(idx)])]
            for idx in range(len(self.bins[region][0]))
        ])
        _mcstat_scale = (1 + np.sqrt(_sumww)/_sumw)**(_mcstat_theta)
        #return _nominal_abs*_nominal_scale*_syst_scale*_mcstat_scale

        # _syst_scale includes the normalisation
        return _nominal_scale*_syst_scale*_mcstat_scale

    def predictions(self, region):
        return sum(
            self.prediction(region, process)
            for process in self.processes[region]
        )

    def poisson_pdfs(self, saturated=False):
        pdf_vals = []
        for r in self.regions:
            pred = np.maximum(0., self.predictions(r))
            if self.asimov:
                if saturated:
                    pred = self.asimov_data[r]
                pdf_vals.append(
                    np.sum(scipy.stats.gamma.logpdf(
                        pred, self.asimov_data[r]+1.,
                    ))
                )
            elif self.toy>=0:
                if saturated:
                    pred = self.toy_data[r]
                pdf_vals.append(
                    np.sum(scipy.stats.poisson.logpmf(
                        self.toy_data[r], pred
                    ))
                )
            else:
                if saturated:
                    pred = self.data[r]
                pdf_vals.append(
                    np.sum(scipy.stats.poisson.logpmf(
                        self.data[r], pred
                    ))
                )
        return np.sum(np.array(pdf_vals))

    def gaussian_pdfs(self):
        return np.sum(scipy.stats.norm.logpdf(np.array([
            self.parameters[n]
            for n in list(set(self.shape_morph_nuisances+self.shape_norm_nuisances))
        ])))

    def gamma_pdfs(self):
        vals = []
        for region in self.regions:
            _sumw = self.mc_proc_sumw[region]
            _sumww = self.mc_proc_sumww[region]
            _neff = self.mc_proc_neff[region]

            _theta = np.array([
                self.parameters["_".join([region, "mcstat", "bin{}".format(idx)])]
                for idx in range(len(self.bins[region][0]))
            ])
            _scale = (1 + np.sqrt(_sumww)/_sumw)**(_theta)
            val = scipy.stats.gamma.logpdf(_scale*_sumw, _neff+1, scale=_sumw/_neff)

            # zero out no data so that it has not effect
            val[_sumww==0.] = np.zeros_like(_sumw[_sumww==0.])
            vals.append(val.sum())

            #for process in self.processes[region]:
            #    _nominal_abs = self.dfmc.loc[(region, process, ""), "sum_w"].values
            #    _neff = self.dfmc.loc[(region, process, ""), :].eval("sum_w**2/sum_ww").values
            #    _theta = self.parameters["_".join([region, process, "mcstat"])]
            #    _sumww = self.dfmc.loc[(region, process, ""), "sum_ww"].values
            #    _scale = (1 + np.sqrt(_sumww)/_nominal_abs)**(_theta)
            #    vals.append(scipy.stats.gamma.logpdf(_scale*_nominal_abs, _neff+1, scale=_nominal_abs/_neff))
            #    #vals.append(scipy.stats.norm.logpdf(_scale*_nominal_abs, loc=_nominal_abs, scale=np.sqrt(_sumww)))
        return np.sum(vals)

    def nll(self, saturated=False):
        val = -2*(
            self.poisson_pdfs(saturated=saturated)
            + self.gaussian_pdfs()
            + self.gamma_pdfs()
        )
        if np.isnan(val):
            raise RuntimeError("NLL is nan")
        if np.isinf(val):
            raise RuntimeError("NLL is inf")
        return val

    def get_minuit_args(self):
        return (self.param_inits,)

    def get_minuit_kwargs(self):
        return {
            "name": self.param_names,
            "fix": self.param_fixes,
            "limit": self.param_limit,
        }

    def create_minuit(self, **kwargs):
        _kwargs = copy.deepcopy(self.get_minuit_kwargs())
        _kwargs.update({"errordef": 1, "pedantic": False})
        _kwargs.update(kwargs)
        return iminuit.Minuit.from_array_func(self, *self.get_minuit_args(), **_kwargs)

    def set_parameters(self, params, init=False):
        if isinstance(params, dict):
            self.parameters = copy.deepcopy(params)
        else:
            for name, val in zip(self.param_names, params):
                self.parameters[name] = val
        if init:
            self.param_inits = [self.parameters[k] for k in self.param_names]

    def __call__(self, params, *args):
        if len(args)>0:
            params = np.array([params]+list(args))
        self.set_parameters(params)
        return self.nll()

    def fit(
        self, fix_all=False, asimov=False, toy=-1, migrad=True, minos=True,
        **kwargs,
    ):
        # asimov dataset - use predictions with the current parameter state
        self.asimov = asimov
        self.asimov_data = {
            r: np.maximum(0., self.predictions(r))
            for r in self.regions
        }

        # toy dataset - use predictions with the current parameter state
        np.random.seed(self.seed+toy)
        self.toy = toy
        self.toy_data = {
            r: scipy.stats.poisson.rvs(np.maximum(1e-10, self.predictions(r)))
            for r in self.regions
        }

        if fix_all:
            self.param_fixes = np.ones_like(self.param_fixes, dtype=bool)

        # iminuit object
        minimizer = self.create_minuit(**kwargs)
        if migrad:
            minimizer.migrad()
        if minos:
            minimizer.minos()
        return minimizer

class NLLModel2(object):
    def __init__(
        self, data, mc, config, same_bin_widths=False,
        shape_type="shape", smooth_region=1.,
        observable_function = "bin_min, bin_max: (bin_min + bin_max)/2."
    ):
        self.verbose = False
        self.dfdata = data
        self.dfmc = mc
        self.config = config

        self.shape_type = shape_type
        self.smooth_region = smooth_region

        self.asimov = False
        self.toy = -1
        self.seed = 123456
        self.saturated = False

        self.norm_nuisances = [
            p["name"]
            for p in config["parameters"]
            if p["constraint"] == "gaussian"
        ]
        self.morph_nuisances = [
            p["name"]
            for p in config["parameters"]
            if p["constraint"] == "gaussian"
        ]

        region_procs = config["regions"]
        self.region_procs = region_procs
        self.regions = list(region_procs.keys())

        self.mcstat_nuisances = {
            r: [p["name"] for p in config["parameters"] if r in p["name"] and p["constraint"]=="gamma"]
            for r in self.regions
        }

        self.param_names = []
        self.param_inits = []
        self.param_fixes = []
        self.param_limit = []
        for idx, params in enumerate(config["parameters"]):
            self.param_names.append(params["name"])
            self.param_inits.append(params["value"])
            self.param_fixes.append(params["fixed"])
            self.param_limit.append(params["limit"])

        self.bins = {}
        self.data = {}
        self.obs = {}
        self.dbins = {}
        for r in self.regions:
            self.data[r] = data.loc[(r,),"sum_w"].values
            bins = data.loc[(r,),:].reset_index()[["bin_min", "bin_max"]].values
            self.obs[r] = eval("lambda "+observable_function)(bins[:,0], bins[:,1])
            self.dbins[r] = bins[:,1] - bins[:,0]
            if same_bin_widths:
                self.dbins[r][:] = 1.
            self.bins[r] = (bins[:,0], bins[:,1])

        # self.data_asimov[r,b]
        # self.data_toy[r,b]

        self.scale_functions = {}
        for r in self.regions:
            self.scale_functions[r] = []
            functions = []
            for p in region_procs[r]:
                self.scale_functions[r].append(
                    config["scale_functions"].get((r, p), "x, w, p: 1.")
                )

        self.sumw = {}
        self.sumww = {}
        for r in self.regions:
            self.sumw[r] = []
            self.sumww[r] = []
            for p in region_procs[r]:
                self.sumw[r].append(mc.loc[(r,p,""),"sum_w"].values)
                self.sumww[r].append(mc.loc[(r,p,""),"sum_ww"].values)
            self.sumw[r] = np.array(self.sumw[r])
            self.sumww[r] = np.array(self.sumww[r])

        self.nom_sum = {}
        self.up_sum = {}
        self.do_sum = {}
        for r in self.regions:
            self.nom_sum[r] = []
            self.up_sum[r] = []
            self.do_sum[r] = []
            for p in region_procs[r]:
                self.nom_sum[r].append(mc.loc[(r,p,""),"sum_w"].sum())

                self.up_sum[r].append([])
                self.do_sum[r].append([])
                for n in self.norm_nuisances:
                    try:
                        self.up_sum[r][-1].append(
                            mc.loc[(r,p,n+"Up"),"sum_w"].sum()
                        )
                        self.do_sum[r][-1].append(
                            mc.loc[(r,p,n+"Down"),"sum_w"].sum()
                        )
                    except KeyError:
                        self.up_sum[r][-1].append(self.nom_sum[r][-1])
                        self.do_sum[r][-1].append(self.nom_sum[r][-1])
            self.nom_sum[r] = np.array(self.nom_sum[r])
            self.up_sum[r] = np.array(self.up_sum[r])
            self.do_sum[r] = np.array(self.do_sum[r])

        self.delta_sum = {}
        self.delta_diff = {}
        for r in self.regions:
            self.delta_sum[r] = []
            self.delta_diff[r] = []
            for p in region_procs[r]:
                self.delta_sum[r].append([])
                self.delta_diff[r].append([])

                nom = mc.loc[(r, p, ""), "sum_w"].values
                nom_norm = nom/(nom*self.dbins[r]).sum()
                for n in self.morph_nuisances:
                    skip = False
                    try:
                        up = mc.loc[(r,p,n+"Up"),"sum_w"].values
                        do = mc.loc[(r,p,n+"Down"),"sum_w"].values
                    except KeyError:
                        delta_sum = np.zeros_like(nom_norm)
                        delta_diff = np.zeros_like(nom_norm)
                        skip = True

                    if not skip:
                        up_norm = up/(up*self.dbins[r]).sum()
                        do_norm = do/(do*self.dbins[r]).sum()

                        if self.shape_type == "shape":
                            up_del = up_norm - nom_norm
                            do_del = do_norm - nom_norm
                        elif self.shape_type == "shapeN":
                            up_del = np.where(
                                (up_norm>0)&(nom_norm>0),
                                np.log(up_norm/nom_norm),
                                np.zeros_like(nom_norm),
                            )
                            do_del = np.where(
                                (do_norm>0)&(nom_norm>0),
                                np.log(do_norm/nom_norm),
                                np.zeros_like(nom_norm),
                            )
                        delta_diff = up_del - do_del
                        delta_sum = up_del + do_del
                    self.delta_diff[r][-1].append(delta_diff)
                    self.delta_sum[r][-1].append(delta_sum)
            self.delta_diff[r] = np.swapaxes(np.array(self.delta_diff[r]), 1, 2)
            self.delta_sum[r] = np.swapaxes(np.array(self.delta_sum[r]), 1, 2)

        self.sumw_procsum = {
            r: (
                mc.loc[(r, pd.IndexSlice[:], ""), "sum_w"]
                .groupby(["bin_min", "bin_max"]).sum().values
            ) for r in self.regions
        }
        self.sumww_procsum = {
            r: (
                mc.loc[(r, pd.IndexSlice[:], ""), "sum_ww"]
                .groupby(["bin_min", "bin_max"]).sum().values
            ) for r in self.regions
        }
        self.neff_procsum = {
            r: self.sumw_procsum[r]**2/self.sumww_procsum[r]
            for r in self.regions
        }

        self.parameters = {}
        self.set_parameters({p["name"]: p["value"] for p in config["parameters"]})

    def _shape_norm(self, region):
        norm = np.ones_like(self.up_sum[region])
        condition = (self.theta_norm>=0.)
        norm[:,condition] = (self.up_sum[region]/self.nom_sum[region][:,np.newaxis])[:,condition]
        norm[:,~condition] = (self.do_sum[region]/self.nom_sum[region][:,np.newaxis])[:,~condition]
        return np.power(norm, np.abs(self.theta_norm[np.newaxis,:]))

    def shape_norm(self, region):
        df = pd.DataFrame(
            self._shape_norm(region),
            columns=self.norm_nuisances,
            index=self.region_procs[region],
        ).stack()
        df.index.names = ["process", "nuisance"]
        df.columns = ["norm"]
        return df

    def _shape_norms(self, region):
        val = self._shape_norm(region)
        if np.isnan(val).any():
            raise RuntimeError("NaN in shape_norm({}): {}".format(region, val))
        if np.isinf(val).any():
            raise RuntimeError("Inf in shape_norm({}): {}".format(region, val))
        return np.prod(self._shape_norm(region), axis=1)

    def shape_norms(self, region):
        df = pd.DataFrame(
            self._shape_norms(region),
            columns=["norm"],
            index=self.region_procs[region],
        )
        df.index.names = ["process"]
        return df

    def _shape_morph(self, region):
        theta_norm = self.theta_morph/self.smooth_region
        theta_norm2 = (theta_norm)**2

        # smooth step function
        step = np.sign(theta_norm)
        condition = (np.abs(theta_norm)<1.)
        step[condition] = (0.125*theta_norm*(theta_norm2*(3.*theta_norm2-10.)+15.))[condition]

        # theta * (diff + sum*smooth_step_function)
        _theta = self.theta_morph[np.newaxis,np.newaxis,:]
        step = step[np.newaxis,np.newaxis,:]
        return 0.5*_theta*(self.delta_diff[region] + self.delta_sum[region]*step)

    def shape_morph(self, region):
        results = self._shape_morph(region)

        df = pd.DataFrame()
        for i1 in range(results.shape[0]):
            for i2 in range(results.shape[2]):
                tdf = pd.DataFrame(results[i1,:,i2], columns=["morph"])
                tdf["process"] = self.region_procs[region][i1]
                tdf["nuisance"] = self.morph_nuisances[i2]
                tdf["bin_min"] = self.bins[region][0]
                tdf["bin_max"] = self.bins[region][1]
                tdf = tdf.set_index(["process", "nuisance", "bin_min", "bin_max"])
                df = pd.concat([df, tdf], axis=0)
        return df

    def _shape_morphs(self, region):
        sumw_norm = np.sum(self.sumw[region]*self.dbins[region][np.newaxis,:], axis=1)[:,np.newaxis]
        shape_vals_norm = self.sumw[region]/sumw_norm
        if self.shape_type=="shapeN":
            shape_vals_norm = np.log(shape_vals_norm)

        shape_vals_norm += np.sum(self._shape_morph(region), axis=2)
        if np.isnan(shape_vals_norm).any():
            raise RuntimeError("NaN in shape_morph({}): {}".format(region, shape_vals_norm))
        if np.isinf(shape_vals_norm).any():
            raise RuntimeError("Inf in shape_morph({}): {}".format(region, shape_vals_norm))

        if self.shape_type=="shapeN":
            shape_vals_norm = np.exp(shape_vals_norm)
        shape_vals_norm = np.maximum(1e-10, shape_vals_norm)

        return shape_vals_norm*sumw_norm #/sumw

    def shape_morphs(self, region):
        results = self._shape_morphs(region)/self.sumw[region]
        df = pd.DataFrame(results.T, columns=self.region_procs[region])
        df["bin_min"] = self.bins[region][0]
        df["bin_max"] = self.bins[region][1]
        df = df.set_index(["bin_min", "bin_max"]).stack()
        df.index.names = ["bin_min", "bin_max", "process"]
        df.columns = ["morph"]
        df = df.reorder_levels(["process", "bin_min", "bin_max"]).sort_index()
        return df

    def _prediction(self, region):
        nom_scale = np.ones_like(self.sumw[region])
        for idx, function in enumerate(self.scale_functions[region]):
            nom_scale[idx,:] = eval('lambda '+function)(
                self.obs[region], self.sumw[region][idx,:], self.parameters,
            )

        syst_scale = self._shape_morphs(region)*(self._shape_norms(region)[:,np.newaxis])

        #mcstat_scale = (1 + np.sqrt(self.sumww)/self.sumw)**(self.theta_mcstat[:,np.newaxis,:])
        mcstat_scale = np.power(
            (1 + np.sqrt(self.sumww_procsum[region])/self.sumw_procsum[region]),
            self.theta_mcstat[region]
        )[np.newaxis,:]

        # return sumw*nom_scale*syst_scale*mcstat_scale
        # syst_scale already includes sumw
        return nom_scale*syst_scale*mcstat_scale

    def prediction(self, region):
        results = self._prediction(region)
        df = pd.DataFrame(results.T, columns=self.region_procs[region])
        df["bin_min"] = self.bins[region][0]
        df["bin_max"] = self.bins[region][1]
        df = df.set_index(["bin_min", "bin_max"]).stack()
        df.index.names = ["bin_min", "bin_max", "process"]
        df.columns = ["prediction"]
        df = df.reorder_levels(["process", "bin_min", "bin_max"]).sort_index()
        return df

    def _predictions(self, region):
        val = self._prediction(region)
        if np.isnan(val).any():
            raise RuntimeError("NaN in prediction({}): {}".format(region, val))
        if np.isinf(val).any():
            raise RuntimeError("Inf in prediction({}): {}".format(region, val))
        return np.sum(self._prediction(region), axis=0)

    def predictions(self, region):
        results = self._predictions(region)
        df = pd.DataFrame(results, columns=["prediction"])
        df["bin_min"] = self.bins[region][0]
        df["bin_max"] = self.bins[region][1]
        df = df.set_index(["bin_min", "bin_max"])
        return df

    def poisson_pdfs(self, region):
        pred = np.maximum(1e-10, self._predictions(region))

        if self.asimov:
            if self.saturated:
                pred = self.data_asimov[region]
            pdfs = scipy.stats.gamma.logpdf(pred, self.data_asimov[region]+1.)
        elif self.toy>=0:
            if self.saturated:
                pred = self.data_toy[region]
            pdfs = scipy.stats.poisson.logpmf(self.data_toy[region], pred)
        else:
            if self.saturated:
                pred = self.data[region]
            pdfs = scipy.stats.poisson.logpmf(self.data[region], pred)
        if np.isnan(pdfs).any():
            raise RuntimeError("NaN in poisson_pdfs({}): {}".format(region, pdfs))
        if np.isinf(pdfs).any():
            raise RuntimeError("Inf in poisson_pdfs({}): {}".format(region, pdfs))
        return np.sum(pdfs)

    def gaussian_pdfs(self):
        pdfs = scipy.stats.norm.logpdf(self.theta_gaus)
        if np.isnan(pdfs).any():
            raise RuntimeError("NaN in gaussian_pdfs({}): {}".format(region, pdfs))
        if np.isinf(pdfs).any():
            raise RuntimeError("Inf in gaussian_pdfs({}): {}".format(region, pdfs))
        return np.sum(pdfs)

    def gamma_pdfs(self, region):
        sumw = self.sumw_procsum[region]
        sumww = self.sumww_procsum[region]
        neff = self.neff_procsum[region]
        theta = self.theta_mcstat[region]

        scale = np.power((1 + np.sqrt(sumww)/sumw), theta)
        pdfs = scipy.stats.gamma.logpdf(scale*neff, neff+1)
        pdfs[sumww==0.] = 0.
        if np.isnan(pdfs).any():
            raise RuntimeError("NaN in gamma_pdfs({}): {}".format(region, pdfs))
        if np.isinf(pdfs).any():
            raise RuntimeError("Inf in gamma_pdfs({}): {}".format(region, pdfs))
        return np.sum(pdfs)

    def nll(self):
        pois = sum(self.poisson_pdfs(r) for r in self.regions)
        gaus = self.gaussian_pdfs()
        gamm = sum(self.gamma_pdfs(r) for r in self.regions)
        val = -2*(pois+gaus+gamm)
        if self.verbose:
            print("pois: {}\ngaus: {}\ngamm: {}\nnll: {}".format(pois, gaus, gamm, val))
        if np.isnan(val):
            raise RuntimeError("NLL is nan")
        if np.isinf(val):
            raise RuntimeError("NLL is inf")
        return val

    def get_minuit_args(self):
        return (self.param_inits,)

    def get_minuit_kwargs(self):
        return dict(
            name=self.param_names,
            fix=self.param_fixes,
            limit=self.param_limit,
        )

    def create_minuit(self, **kwargs):
        _kwargs = copy.deepcopy(self.get_minuit_kwargs())
        _kwargs.update({"errordef": 1, "pedantic": False})
        _kwargs.update(kwargs)
        return iminuit.Minuit.from_array_func(
            self, *self.get_minuit_args(), **_kwargs,
        )

    def set_parameters(self, params, init=False):
        if isinstance(params, dict):
            params = np.array([
                params.get(name, init)
                for init, name in zip(self.param_inits, self.param_names)
            ])

        if np.isnan(params).any():
            raise RuntimeError(
                "NaN in input parameters. Previous values:\n{}\nNew values:\n{}".format(
                    self.parameters, params,
                )
            )
        if np.isinf(params).any():
            raise RuntimeError(
                "Inf in input parameters. Previous values:\n{}\nNew values:\n{}".format(
                    self.parameters, params,
                )
            )

        for name, val in zip(self.param_names, params):
            self.parameters[name] = val

        self.theta_mcstat = {}
        for r in self.regions:
            self.theta_mcstat[r] = []
            for name in self.mcstat_nuisances[r]:
                self.theta_mcstat[r].append(self.parameters[name])
            self.theta_mcstat[r] = np.array(self.theta_mcstat[r])

        self.theta_gaus = []
        self.theta_norm = []
        self.theta_morph = []
        for name in set(self.norm_nuisances+self.morph_nuisances):
            self.theta_gaus.append(self.parameters[name])
        for name in self.norm_nuisances:
            self.theta_norm.append(self.parameters[name])
        for name in self.morph_nuisances:
            self.theta_morph.append(self.parameters[name])
        self.theta_gaus = np.array(self.theta_gaus)
        self.theta_norm = np.array(self.theta_norm)
        self.theta_morph = np.array(self.theta_morph)

        if init:
            self.param_inits[:] = params

    def __call__(self, params, *args):
        if len(args)>0:
            params = np.array([params]+list(args))
        self.set_parameters(params)
        return self.nll()

    def fit(self, asimov=False, toy=-1, migrad=False, minos=False, **kwargs):
        self.asimov = asimov
        self.data_asimov = {
            r: np.maximum(0., self._predictions(r))
            for r in self.regions
        }

        np.random.seed(self.seed+toy)
        self.toy = toy
        self.data_toy = {
            r: scipy.stats.poisson.rvs(self._predictions(r))
            for r in self.regions
        }

        minimizer = self.create_minuit(**kwargs)
        if migrad:
            minimizer.migrad()
        if minos:
            minimizer.minos()
        self.minimizer = minimizer
        return minimizer

    def draw_data(self, region, ax, ratio=False, pull=False, **kwargs):
        data = self.data[region].copy()
        lower, upper = poisson_interval(data)
        if not pull:
            if ratio:
                mc = self.predictions(region)["prediction"].copy()
                data /= mc
                lower /= mc
                upper /= mc

            kw = dict(fmt='o', color='black')
            kw.update(kwargs)
            ax.errorbar(
                sum(self.bins[region])/2., data,
                yerr=(data-lower, upper-data), **kw,
            )
        else:
            mc = self.predictions(region)["prediction"].copy()
            mclower, mcupper = confidence_interval(self, self.minimizer, "_predictions", region)
            diff = data - mc
            sigma = np.sqrt((data-lower)**2+(mc-mcupper)**2)
            sigma[diff<0.] = np.sqrt((data-upper)**2+(mc-lower)**2)[diff<0.]

            kw = dict(color='black')
            kw.update(kwargs)
            ax.plot(sum(self.bins[region])/2., diff/sigma, 'o', **kw)

    def draw_prediction_total(
        self, region, ax, ratio=False, prefit=False, band=True, hkwargs={},
        fkwargs={},
    ):
        if prefit:
            saved_params = copy.deepcopy(self.parameters)
            self.set_parameters(self.param_inits)
            mc_prefit = self.predictions(region)["prediction"].copy()
            self.set_parameters(saved_params)
        mc = self.predictions(region)["prediction"].copy()
        lower, upper = confidence_interval(self, self.minimizer, "_predictions", region)
        if ratio:
            lower /= mc
            upper /= mc
            if prefit:
                mc_prefit /= mc
                mc = mc_prefit
            else:
                mc = np.ones_like(mc)

        kw = dict(histtype='step', color='black')
        if prefit:
            kw["ls"] = "--"
        kw.update(hkwargs)
        ax.hist(
            self.bins[region][0],
            bins=list(self.bins[region][0])+[self.bins[region][1][-1]],
            weights=mc, **kw,
        )

        if band:
            kw = dict(step='post', color='#d9d9d9')
            kw.update(fkwargs)
            ax.fill_between(
                list(self.bins[region][0])+[self.bins[region][1][-1]],
                list(lower)+[lower[-1]], list(upper)+[upper[-1]], **kw,
            )

    def draw_prediction_procs(self, region, ax, **kwargs):
        mc_procs = self.prediction(region).copy()
        procs = self.region_procs[region][::-1]

        kw = dict(
            stacked=True, color=[process_colours.get(p, 'blue') for p in procs],
            label=[process_names.get(p, p) for p in procs],
        )
        kw.update(kwargs)

        #w = mc_procs.reorder_levels(["bin_min", "bin_max", "process"])
        ws = [mc_procs.loc[(p,)] for p in procs]
        ax.hist(
            [self.bins[region][0]]*len(procs),
            bins=list(self.bins[region][0])+[self.bins[region][1][-1]],
            weights=ws, **kw,
        )

def run_minos(
    params, nll_args=tuple(), nll_kwargs={}, migrad_kwargs={"ncall": 100000},
    pysge_function="local_submit", pysge_args=tuple(), pysge_kwargs={},
    params_guess=None,
):
    def run_minos_param(
        _param, _nll_args, _nll_kwargs, _migrad_kwargs,
        _param_guess=None,
    ):
        mod_= NLLModel2(*_nll_args, **_nll_kwargs)
        if _param_guess is not None:
            mod_.set_parameters(_param_guess, init=True)
        mini_ = mod_.fit(migrad=False, minos=False)
        mini_.migrad(**_migrad_kwargs)
        mini_.minos(_param)
        return {
            "parameter": _param,
            "value": mini_.values[_param],
            "merror_up": mini_.merrors[(_param, 1.)],
            "merror_down": mini_.merrors[(_param, -1.)],
        }

    tasks = [{
        "task": run_minos_param,
        "args": (p, nll_args, nll_kwargs, migrad_kwargs),
        "kwargs": {"_param_guess": params_guess},
    } for p in params]

    results = getattr(pysge, pysge_function)(
        *tuple([tasks]+list(pysge_args)),
        **pysge_kwargs,
    )
    return pd.DataFrame(results)

def run_impacts(
    params, poi, nll_args=tuple(), nll_kwargs={},
    fit_kwargs={"migrad": False, "minos": False},
    migrad_kwargs={"ncall": 100000}, pysge_function="local_submit",
    pysge_args=tuple(), pysge_kwargs={}, params_guess=None,
    return_tasks=False,
):
    def run_impact_param(
        _param, _poi, _nll_args, _nll_kwargs, _fit_kwargs, _migrad_kwargs,
        _param_guess=None,
    ):
        mod_= NLLModel2(*_nll_args, **_nll_kwargs)
        if _param_guess is not None:
            mod_.set_parameters(_param_guess, init=True)
        mod_params = copy.deepcopy(mod_.parameters)
        mini_ = mod_.fit(**_fit_kwargs)
        mini_.migrad(**_migrad_kwargs)
        valid_nom = mini_.migrad_ok()
        mini_.minos(_param)

        if _param == _poi:
            return {
                "param": _param,
                "param_value": mini_.values[_param],
                "param_merrup": mini_.merrors[(_param, 1.)],
                "param_merrdown": mini_.merrors[(_param, -1.)],
                "poi": _poi,
                "poi_paramup": mini_.merrors[(_param, 1.)],
                "poi_paramdown": mini_.merrors[(_param, -1.)],
            }

        poi_bf_central = mini_.values[_poi]
        param_bf_central = mini_.values[_param]
        param_bf_up = mini_.merrors[(_param, 1.)]
        param_bf_down = mini_.merrors[(_param, -1.)]

        # Perform fit at +1 sigma
        mod_params[_param] = param_bf_central + param_bf_up
        mod_.set_parameters(mod_params, init=True)
        miniup_ = mod_.fit(**_fit_kwargs)
        miniup_.fixed[_param] = True
        miniup_.migrad(**_migrad_kwargs)
        valid_up = miniup_.migrad_ok()
        poi_bf_nuisance_up = miniup_.values[_poi]

        # Perform fit at -1 sigma
        mod_params[_param] = param_bf_central + param_bf_down
        mod_.set_parameters(mod_params, init=True)
        minido_ = mod_.fit(**_fit_kwargs)
        minido_.fixed[_param] = True
        minido_.migrad(**_migrad_kwargs)
        valid_do = minido_.migrad_ok()
        poi_bf_nuisance_down = minido_.values[_poi]

        return {
            "param": _param,
            "param_value": param_bf_central,
            "param_merrup": param_bf_up,
            "param_merrdown": param_bf_down,
            "poi": _poi,
            "poi_paramup": poi_bf_nuisance_up-poi_bf_central,
            "poi_paramdown": poi_bf_nuisance_down-poi_bf_central,
            "valid": valid_nom and valid_up and valid_do,
        }

    tasks = [{
        "task": run_impact_param,
        "args": (p, poi, nll_args, nll_kwargs, fit_kwargs, migrad_kwargs),
        "kwargs": {"_param_guess": params_guess},
    } for p in params]
    if return_tasks:
        return tasks

    results = getattr(pysge, pysge_function)(
        *tuple([tasks]+list(pysge_args)),
        **pysge_kwargs,
    )
    return pd.DataFrame(results)

def confidence_interval(model, minimizer, attr, *args, **kwargs):
    def pred_func(params):
        model.set_parameters(params)
        return getattr(model, attr)(*args, **kwargs)

    bf_params = minimizer.np_values()

    cov = minimizer.np_matrix(skip_fixed=False)
    jac = numdifftools.Jacobian(pred_func)(bf_params)
    err = scipy.stats.chi2.ppf(0.68, df=1)*np.sqrt(
        np.matmul(jac, np.matmul(cov, jac.T)).diagonal()
    )
    cent = pred_func(bf_params)
    return cent-err, cent+err
