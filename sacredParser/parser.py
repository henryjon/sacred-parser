import inspect
import itertools
import json
import os
import pickle
import re
import warnings

import pandas as pd
from sacredParser import core


class FileStorageParser:
    """Parser for the sacred FileStorageObserver"""

    def __init__(self, basedir, processed_dirname=None):
        self.basedir = basedir
        self.processed_dirname = processed_dirname
        self.eids = [
            int(f)
            for f in os.listdir(self.basedir)
            if os.path.isdir(os.path.join(self.basedir, f))
            and re.search("[0-9]+", f) is not None
        ]

        assert len(self.eids) > 0, "No experiment outputs found"
        self.eids.sort()

        # Config files
        self.cfg_df = self._config_df()
        self.artifacts = self._artifacts()

    def _config_df(self):
        cfgs = [None] * len(self.eids)

        for i, eid in enumerate(self.eids):
            with open(os.path.join(self._eid_dir(eid), "config.json")) as f:
                cfg = json.load(f)

            if self.processed_dirname is None:
                assert "eid" not in cfg, '"eid" should not be a config field'

            cfg = core.flatten_json(cfg)
            cfg["eid"] = eid

            cfgs[i] = cfg

        cfg_df = pd.DataFrame(cfgs).set_index("eid")
        cfg_df.columns = cfg_df.columns.str.lower().str.replace("[^a-z0-9]", "_")

        return cfg_df

    def _artifacts(self):
        artifact_filenames = set(
            f
            for eid in self.eids
            for f in os.listdir(self._eid_dir(eid))
            if os.path.isfile(os.path.join(self._eid_dir(eid), f))
            and (f not in ["config.json", "metrics.json", "run.json", "cout.txt"])
        )
        return {
            os.path.splitext(f)[0]: {eid: self._load_file(eid, f) for eid in self.eids}
            for f in artifact_filenames
        }

    def _argnames(self):
        return (
            ["eid"] + [str(name) for name in self.cfg_df.columns] + list(self.artifacts)
        )

    def _eid_dir(self, eid, dirname=None):
        root = os.path.join(self.basedir, str(eid))

        if dirname is not None:
            return os.path.join(root, dirname)

        if self.processed_dirname is not None:
            return os.path.join(root, self.processed_dirname)

        return root

    def _load_file(self, eid, filename):
        ext = os.path.splitext(filename)[1]
        path = os.path.join(self._eid_dir(eid), filename)

        if not os.path.isfile(path):
            return None
        if ext == ".feather":
            return pd.read_feather(path)
        if ext == ".json":
            with open(path) as f:
                return json.load(f)
        if ext in [".pickle", ".pkl"]:
            with open(path, "rb") as f:
                return pickle.load(f)

        raise NotImplementedError(ext)

    def _kwargs(self, argnames, eid):
        return {
            arg: eid
            if arg == "eid"
            else self.cfg_df[arg][eid]
            if arg in self.cfg_df.columns
            else self.artifacts[arg][eid]
            if arg in self.artifacts
            else None
            for arg in argnames
            if arg in self._argnames()
        }

    def eid_match(self, **kwargs):
        ix = True
        for k, v in kwargs.items():
            ix = ix & (self.cfg_df[k] == v)

        return self.cfg_df.index[ix]

    def add_artifact(self, name, f):
        if name == "eid":
            raise Exception('Cannot use name "eid"')

        if name in self.cfg_df.columns:
            raise Exception(f"Name {name} already a config hyperparameter name")

        argnames = inspect.getfullargspec(f)[0]
        self.artifacts[name] = {
            eid: f(**self._kwargs(argnames, eid)) for eid in self.eids
        }

    def add_cfg_col(self, name, f):
        if name == "eid":
            raise Exception('Cannot use name "eid"')

        if name in self.artifacts:
            raise Exception("Name {} already an artifact name".format(name))

        if name in self.cfg_df.columns:
            warnings.warn("You are overwriting a config column")

        argnames = inspect.getfullargspec(f)[0]
        self.cfg_df[name] = [f(**self._kwargs(argnames, eid)) for eid in self.eids]

    def do_for_eid(self, eid, f, **kwargs):
        """Apply the function f to a single eid"""
        argnames = [arg for arg in inspect.getfullargspec(f)[0] if arg not in kwargs]
        return f(**self._kwargs(argnames, eid), **kwargs)

    def do_along_eids(self, f, **kwargs):
        """Apply the function f to each eid"""
        argnames = [arg for arg in inspect.getfullargspec(f)[0] if arg not in kwargs]
        return [f(**self._kwargs(argnames, eid), **kwargs) for eid in self.eids]

    def unnested_artifact(self, artifact_name, eids=None):
        df = (
            pd.concat(
                [
                    artifact.assign(eid=eid)
                    for eid, artifact in self.artifacts[artifact_name].items()
                    if (eids is None) or (eid in eids)
                ]
            )
            .set_index("eid")
            .join(self.cfg_df)
        ).reset_index()
        return df

    def _save_artifact(self, artifact_name, eid, dirname):
        x = self.artifacts[artifact_name][eid]
        root = self._eid_dir(eid, dirname=dirname)

        if x is None:
            return None

        if type(x) is pd.DataFrame:
            path = os.path.join(root, "{}.pkl".format(artifact_name))
            return x.to_pickle(path)

        if type(x) is list:
            path = os.path.join(root, "{}.json".format(artifact_name))
            with open(path, "w") as f:
                out = json.dump(x, f, cls=core.NumpyEncoder)
            return out

        raise NotImplementedError(type(x))

    def _save_config(self, eid, dirname):
        root = self._eid_dir(eid, dirname=dirname)
        if not os.path.isdir(root):
            os.mkdir(root)

        path = os.path.join(root, "config.json")

        with open(path, "w") as f:
            json.dump(dict(self.cfg_df.loc[eid]), f, cls=core.NumpyEncoder)

    def save(self, dirname):
        """Save a (likely processed) experiment: config and artifacts"""

        for eid in self.eids:
            self._save_config(eid, dirname=dirname)

            for artifact_name in self.artifacts:
                self._save_artifact(artifact_name, eid, dirname=dirname)
