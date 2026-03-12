import re
import numpy as np
import os
import glob
import warnings
from copy import deepcopy
import json
from extraction_utils import *


ANALOG_TRIGGERS_MAPPING = {"Main": 0, "Playback": 1}
DIGITAL_TRIGGERS_MAPPING = {"Basler": 0, "Sounds": 1}


class SequenceTT(object):
    """

    """
    def __init__(self, folder=None, n_iter=None):
        self.container = dict()
        self.keys = list()
        self.order = np.empty(0, dtype=int)
        self.numbers = np.empty(0, dtype=int)
        self.total_iter = 0
        self.allowed = ["playback", "tracking", "warmup", "warmdown", "mock"]
        if n_iter is not None:
            self.total_iter = n_iter
        self.recording_length = 0

        if folder is not None:
            self._load(folder)

    def _load(self, folder):
        d = np.load(os.path.join(folder, "tt.npz"), allow_pickle=True)
        self.recording_length = d["recording_length"][0]
        self.order = d["order"]
        self.total_iter = d["n_iter"][0]
        self.keys = [key.decode() for key in d["keys"]]
        self.numbers = d["numbers"]
        for i, key in enumerate(self.keys):
            tones, triggers = d[key][0], d[key][1]
            self.container[key] = Pair(tones, triggers, type_of=get_type_from_pattern(key), order=self.order[i])

    def get_number_iteration(self):
        return self.total_iter

    def save(self, folder, fn=None):
        """

        """
        if fn is None:
            fn = "tt.npz"
        fn = os.path.join(folder, fn)
        kwargs = dict()
        kwargs["order"] = np.array(self.order)
        kwargs["n_iter"] = np.array([self.total_iter])
        kwargs["recording_length"] = np.array([self.recording_length])
        kwargs["keys"] = self._build_chararray()
        kwargs["numbers"] = self.numbers
        for key in self.container.keys():
            kwargs[key] = self.container[key].get_stacked()
        # kwargs = {key: self.container[key].get_stacked() for key in self.container.keys()}
        np.savez(fn, **kwargs)

    def _build_chararray(self):
        n = np.array(self.keys).shape
        ch = np.chararray(n, itemsize=5)
        for i, elt in enumerate(self.keys):
            ch[i] = elt
        return ch

    def get_container(self):
        return self.container

    def get_xp_type_all(self, type_of, as_tt=True):
        """
        On va chercher toutes les expériences d'un certain type.
        """
        assert (type_of in self.allowed), "Wrong type..."
        pattern = get_pattern_from_type(type_of)
        if as_tt:
            out = SequenceTT()
            for i, k in enumerate(self.keys):
                if re.search(pattern, k):
                    tmp = self.container[k]
                    p = Pair(tmp.tones, tmp.triggers, type_of, number=self.numbers[i], order=self.order[i])
                    out.add(p)
            out.set_n_iter(self.total_iter)
        else:
            out = dict()
            for k in self.container.keys():
                if re.search(pattern, k):
                    out[k] = self.container[k]
        return out

    def merge(self, type_of):
        """
        On lui donne un type d'expériences. Renvoie une paire. Mets touts les triggers et les tones dedans.
        """
        d_out = self.get_xp_type_all(type_of, as_tt=False)
        l_number = list()
        # mettre dans l'ordre
        for k in d_out.keys():
            l_number.append(k)
        l_number.sort()
        tones = list()
        triggers = list()
        for k in l_number:
            p = d_out[k]
            type_of = p.get_type()
            tones.append(p.get_tones())
            triggers.append(p.get_triggers())
        triggers = np.hstack(triggers)
        tones = np.hstack(tones)
        return Pair(tones, triggers, type_of)

    def get_xp_number(self, type_of, n):
        """
        On demande une expérience d'un type donné, à un moment donné.
        """
        assert (type_of in self.allowed), "Wrong type..."
        if type_of not in ["warmup", "warmdown"]:
            assert (n < self.total_iter), "Unavailable."
        pattern = get_pattern_from_type(type_of) + str(n)
        assert (pattern in self.keys), "Not existing"
        return self.container[pattern]

    def get_all_number(self, n):
        """
        On va chercher le triplet Playback, Tracking, Mock.
        """
        assert (n < self.total_iter), "Unavailable."
        pattern = str(n)
        d_out = dict()
        for k in self.container.keys():
            if re.search(pattern, k):
                d_out[k] = self.container[k]

        return d_out

    def get_from_type_and_number(self, type_of, n):
        assert (type_of in self.allowed), "Wrong type..."
        assert (n < self.total_iter), "Unavailable."
        pattern = get_pattern_from_type(type_of) + str(n)
        for k in self.container.keys():
            if re.search(pattern, k):
                return self.container[k]

    def get_triggers(self):
        list_triggers = list()
        for elt in self.keys:
            list_triggers.append(self.container[elt].get_triggers())
        return np.hstack(list_triggers)

    def get_all_triggers_for_type(self, type_of):
        p = self.merge(type_of)  # sort un objet Pair.
        return p.get_triggers()

    def add(self, pairs):
        pattern = pairs.get_pattern()
        order = pairs.order
        number = pairs.number
        assert (pattern not in self.keys), "Already in DataStructure."
        assert (order not in self.order), "Already in DataStructure."
        self.numbers = np.hstack((self.numbers, number))
        self.order = np.hstack((self.order, order))
        self.keys.append(pattern)
        self.container[pattern] = pairs

    def set_recording_length(self, length):
        self.recording_length = length

    def get_recording_length(self):
        return self.recording_length

    def set_n_iter(self, n_iter):
        self.total_iter = n_iter

    def get_n_iter(self):
        return self.total_iter

    def get_borders(self):
        d = dict()
        d_tr = dict()
        d_pb = dict()
        begin, end = self.get_xp_number("warmup", 0).get_begin_and_end_triggers()
        d["warmup"] = [begin, end]
        begin, end = self.get_xp_number("warmdown", 0).get_begin_and_end_triggers()
        d["warmdown"] = [begin, end]
        for i in range(self.total_iter):
            tr = self.get_xp_number("tracking", i)
            pb = self.get_xp_number("playback", i)

            begin, end = tr.get_begin_and_end_triggers()

            if i == 0:
                begin = pb.triggers[0] - 5 * 30000 * 60
            d_tr[i] = [begin, end]

            if i < self.total_iter - 1:
                tr = self.get_xp_number("tracking", i + 1)
            else:
                tr = d["warmdown"][0]
            begin, end = pb.triggers[0], tr.triggers[0]
            d_pb[i] = [begin, end]

        d["tracking"] = d_tr
        d["playback"] = d_pb
        # d[""]
        return d


class TT(object):
    def __init__(self, tones, triggers):
        assert (len(tones) == len(triggers)), "Tones and Triggers have different length."
        self.length = len(tones)
        self.tones = tones
        self.triggers = triggers

    def get_outside_bandwidth(self, bandwidth):
        """
        Retourne les paires triggers / fréquences qui correspondent à des fréquences en dehors de la bande passante
        d'un cluster.
        """
        idx = np.logical_and(self.tones >= bandwidth[0], self.tones <= bandwidth[1])
        tt_cleaned = TT(tones=self.tones[~idx], triggers=self.triggers[~idx])
        return tt_cleaned

    def get_inside_bandwidth(self, bandwidth):
        """
        Retourne les paires triggers / fréquences qui correspondent à des fréquences dans de la bande passante
        d'un cluster.
        """
        idx = np.logical_and(self.tones >= bandwidth[0], self.tones <= bandwidth[1])
        tt_cleaned = TT(tones=self.tones[idx], triggers=self.triggers[idx])
        return tt_cleaned

    def get_for_tone(self, tone):
        """
        On demande l'objet TT pour une fréquence donnée.
        """
        idx = np.equal(self.tones, tone)
        tt_cleaned = TT(tones=self.tones[idx], triggers=self.triggers[idx])
        return tt_cleaned


class Pair(object):
    def __init__(self, tones, triggers, type_of, number=None, order=None):
        assert (len(tones) == len(triggers)), "Tones and Triggers have different length."
        self.tones = tones

        self.triggers = triggers
        self.tt = TT(tones, triggers)

        assert (type_of in ["playback", "tracking", "warmup", "warmdown", "mock", "PureTones", "silence"]), "Wrong type..."
        self.type = type_of

        if order is not None:
            self.order = order
        else:
            self.order = None

        if number is not None:
            self.number = number
            self.pattern = get_pattern_from_type(self.type) + str(self.number)
        else:
            self.number = None
            self.pattern = None

    def get_stacked(self):
        return np.vstack((self.tones, self.triggers))

    def get_tones(self):
        return self.tones

    def get_triggers(self):
        return self.triggers

    def get_pairs(self):
        return self.tt

    def get_pattern(self):
        return self.pattern

    def get_type(self):
        return self.type

    def get_begin_and_end_triggers(self):
        return self.triggers[0], self.triggers[-1]


class XPSingleton(object):
    """
    Bout de session avant processing
    """
    def __init__(self, type_of, order, number, duration, tones, fs=30e3):
        self.t = duration * 60 * fs
        self.order = order
        self.n = number
        assert(type_of in ["playback", "tracking", "warmup", "warmdown", "mock"]), "Wrong type..."
        self.type = type_of
        if self.type == "warmup":
            self.tag = -1
        elif self.type == "warmdown":
            self.tag = -2
        elif self.type == "tracking":
            self.tag = 0 + self.n
        elif self.type == "mock":
            self.tag = 10 + self.n
        x = append_zero(self.n, 10)
        self.pattern = get_pattern_from_type(self.type) + str(x)
        self.tones = tones


class Sequence(object):
    """
    Regroupe les xp d'une session.
    """
    def __init__(self):
        self.container = dict()
        self.order = list()
        self.patterns = list()
        self.duration = dict()

    def get_n_tones_for(self, type_of):
        l_out = self.get_all_xp_for_type(type_of)
        s = 0
        for elt in l_out:
            s += len(elt.tones)
        return s

    def get_all_xp_for_type(self, type_of):
        assert (type_of in ["playback", "tracking", "warmup", "warmdown", "mock"]), "Wrong type..."
        pattern = get_pattern_from_type(type_of)
        l_order = list()
        l_out = list()
        for k in self.container.keys():
            if re.search(pattern, k):
                l_order.append(k)
        l_order.sort()
        for k in l_order:
            l_out.append(self.container[k])
        return l_out

    def get_all_number(self, n):
        """
        On va chercher le triplet Playback, Tracking, Mock.
        """
        # assert (n < self.total_iter), "Unavailable."
        pattern = append_zero(n, 10)
        l_type = ["tracking", "playback", "mock"]
        d_out = {key: get_pattern_from_type(key) + pattern for key in l_type}
        for k in self.container.keys():
            for key in d_out.keys():
                if d_out[key] == k:
                    d_out[key] = self.container[k]

        return d_out

    def get_xp_number(self, type_of, n):
        out = self.get_all_xp_for_type(type_of)
        # todo: assert le numéro est OK.
        return out[n]

    def add(self, xp):
        pattern = xp.pattern
        order = xp.order
        assert (pattern not in list(self.container.keys())), "Already in Sequence."
        assert (order not in self.order), "Already in Sequence."
        self.patterns.append(pattern)
        self.order.append(order)
        type_of = xp.type
        if type_of not in list(self.duration.keys()):
            self.duration[type_of] = xp.t
        self.container[pattern] = xp

    def get_duration_for(self, type_of):
        assert (type_of in ["playback", "PLAYBACK", "tracking", "warmup", "warmdown", "mock"]), "Wrong type..."
        return self.duration[type_of]

    def get_for_types(self, types):
        order = np.array(self.order)
        arg_order = np.argsort(order)
        patterns = [self.patterns[elt] for elt in arg_order]
        assert(type(types) == list or type(types) == str), "Wrong type for types. str or list are required."
        keep = list()
        if type(types) == str:
            pattern_to_search = [get_pattern_from_type(types)]
        else:
            pattern_to_search = [get_pattern_from_type(type_of) for type_of in types]
        for elt in patterns:
            for p in pattern_to_search:
                if re.search(p, elt):
                    keep.append(self.container[elt])
        return keep

    def get_in_order(self, pb=False):
        order = np.array(self.order)
        arg_order = np.argsort(order)
        patterns = [self.patterns[elt] for elt in arg_order]
        keep = list()
        pattern = get_pattern_from_type("playback")
        for elt in patterns:
            if pb:
                if re.search(pattern, elt):
                    keep.append(elt)
            else:
                if not re.search(pattern, elt):
                    keep.append(elt)

        for i, elt in enumerate(keep):
            keep[i] = self.container[elt]
        return keep

    def get_all_tones_for(self, type_of):
        keep = self.get_in_order_for_type(type_of=type_of)
        out = np.hstack([xp.tones for xp in keep])
        return out

    def get_in_order_for_type(self, type_of):
        order = np.array(self.order)
        arg_order = np.argsort(order)
        patterns = [self.patterns[elt] for elt in arg_order]
        keep = list()
        pattern = get_pattern_from_type(type_of)
        for elt in patterns:
            if re.search(pattern, elt):
                keep.append(elt)
        for i, elt in enumerate(keep):
            keep[i] = self.container[elt]
        return keep

    def get_tracking(self):
        order = np.array(self.order)
        arg_order = np.argsort(order)
        patterns = [self.patterns[elt] for elt in arg_order]
        keep = list()
        pattern = [get_pattern_from_type("playback"), get_pattern_from_type("mock")]

        for elt in patterns:
            if not re.search(pattern[0], elt) and not re.search(pattern[1], elt):
                keep.append(elt)

        for i, elt in enumerate(keep):
            keep[i] = self.container[elt]
        return keep


def read_json(folder):

    key_to_fetch = "Experiment_1"  # on cherche la première expérience. ?

    out = list(glob.glob(os.path.join(folder, "session_*.json")))

    exp_type = ""

    try:
        assert (len(out) == 1), "Glob in folder should be of length 1."
        fname = out[0]
        with open(fname, "r") as f:
            d = json.load(f)

    except AssertionError as error:
        print("Error: ", error)

    try:
        assert (key_to_fetch in d.keys()), f"{key_to_fetch} not in json."
        sub_d = d[key_to_fetch]
        exp_type = sub_d["Type"]

    except AssertionError as error:
        print("Error: ", error)

    # is_v2(d) => utiliser ça.
    version_key = "Version"  # Là on demande si c'est le nouveau format de fichier.
    # todo : la nouvelle fonction doit retourner un json.
    if version_key in d.keys():
        pass

    else:
        pass

    return exp_type


def is_v2(d):
    version_key = "Version"  # Là on demande si c'est le nouveau format de fichier.
    if version_key in d.keys():
        return True

    else:
        return False
    


def extract_v2(d, triggers, folder):
    seq, length = check_already_extracted(folder)
        #exp_type = read_json(folder)
    exp_type = "Playback"
    seq = extract_according_exp_type(exp_type, triggers=triggers, folder=folder, compatibility=True)

    seq.set_recording_length(length)
    seq.save(folder)
    print("All izz well")


def extract_data(folder, analog_channels=None, digital_channels=None, compatibility=False):

    seq, length = check_already_extracted(folder)

    if seq and length:
        return extract_tt(folder)

    else:
        dig_file = os.path.join(folder, "dig_in.npy")
        analog_file = os.path.join(folder, "analog_in.npy")

        # todo: faire fonction pour le calcul de la longueur de l'enregistrement.
        if length:
            length = get_recording_length(folder)
        else:
            assert (os.path.exists(dig_file)), "No digital triggers file in directory."
            d_trigs = np.load(os.path.join(folder, "dig_in.npy"))
            length = d_trigs.shape[1]
            save_recording_length(folder, length)

        # FIN de la fonction à créer.
        if seq:
            seq = extract_tt(folder)
        else:
            triggers = dict()
            n_dig_files, l_dig_files = check_digital_triggers(folder)
            if n_dig_files > 0:
                out_digital = load_digital_files(l_dig_files)

            else:
                assert(os.path.exists(dig_file)), "No digital triggers file in directory."
                out_digital = process_digital_file(folder)

            if digital_channels is None:
                triggers["dig"] = out_digital["sounds"]

            n_analog_files, l_analog_files = check_analog_triggers(folder)

            if n_analog_files > 0:
                out_analog = load_analog_files(l_analog_files)

            else:
                assert(os.path.exists(analog_file)), "No analog triggers file in directory."
                out_analog = process_analog_file(folder, compatibility=compatibility)

            if analog_channels is None:
                triggers["tracking"] = out_analog["tracking"]
                if not compatibility:
                    triggers["mock"] = out_analog["mock"]
                triggers["playback"] = out_analog["playback"]

            exp_type = read_json(folder)
            seq = extract_according_exp_type(exp_type, triggers=triggers, folder=folder, compatibility=compatibility)

        seq.set_recording_length(length)
        seq.save(folder)
        return seq


def extract_according_exp_type(type_of, triggers, folder, compatibility=False):
    """

    """
    seq = None

    if type_of == "Playback":
        if compatibility:
            seq = get_data_4(triggers, folder)
        else:
            seq = get_playback(triggers, folder)

    elif type_of == "PureTones":
        seq = None  # get_tonotopy(folder=folder, triggers=triggers)

    return seq


# def get_tonotopy(triggers, folder):
#     """
#     Charge une expérience de tonotopie.
#     """
#     tt_seq = Tonotopy(n_iter=1)
#
#     tones = np.empty(0)
#
#     for file in glob.glob(os.path.join(folder, "tones_*.bin")):
#         tones = np.hstack((tones, np.fromfile(file, dtype=np.double)))
#
#     tt_seq.add(Pair(tones, triggers["tracking"], "PureTones", 0))
#
#     return tt_seq


def get_playback(triggers, folder):
    """
    Charge une expérience Playback.
    """
    l_tracking, l_mock, l_pb, l_warmup = fetch_tones(folder)

    # Cette assertion peut mener à des erreurs.
    assert (len(l_pb) == len(l_tracking) == len(l_mock))

    n_iter = len(l_pb)

    fn = os.path.join(folder, "durations.json")

    # todo : penser à incorporer les données dans le .json de manip.
    if os.path.exists(fn):
        with open(fn, "r") as f:
            d = json.load(f)
        duration_tr = d["tracking"]
        duration_warmup = d["warmup"]
        duration_warmdown = d["warmdown"]

    else:
        duration_tr = 5
        duration_warmup = 10
        duration_warmdown = 10

    c = 0
    sequence = Sequence()
    sequence.add(XPSingleton("warmup", c, 0, duration_warmup, tones=l_warmup[0]))
    c += 1
    for i in range(n_iter):
        sequence.add(XPSingleton("tracking", c, i, duration_tr, tones=l_tracking[i]))
        c += 1
        sequence.add(XPSingleton("mock", c, i, duration_tr, tones=l_mock[i]))
        c += 1
        sequence.add(XPSingleton("playback", c, i, duration_tr, tones=l_pb[i]))
        c += 1

    sequence.add(XPSingleton("warmdown", c, 0, duration_warmdown, tones=l_warmup[1]))

    d_out = divide_triggers_2(triggers, sequence, n_iter=n_iter)
    return d_out


def get_bin_pos(t_0, _t, _size):
    _d = _t - t_0
    _p = int(_d / _size)
    return _p


def iterate_tones_folder(folder, pattern):
    """
    Retourne une liste de np.ndarray avec les tons joués.
    """
    seq_out = list()
    _glob = glob.glob(os.path.join(folder, pattern))
    _glob = list(_glob)
    _glob.sort()
    for file in _glob:
        seq_out.append(np.fromfile(file, dtype=np.double))
    return seq_out


def if_complete_2(analog, sequence, tt, type_of=None):
    if type_of is None:
        xp_list = sequence.get_tracking()
    else:
        xp_list = sequence.get_in_order_for_type(type_of)
    for elt in xp_list:
        t = elt.tones
        triggers, analog = analog[:len(t)], analog[len(t):]
        tt.add(Pair(t, triggers, elt.type, number=elt.n, order=elt.order))
    return tt


def build_pair_from_singleton(analog, tones, singleton):
    p = Pair(tones, analog, singleton.type, number=singleton.n, order=singleton.order)
    return p


def divide_triggers_2(triggers, sequence, n_iter):
    tt_seq = SequenceTT(n_iter=n_iter)
    tt_seq = triggers_tones_inspection_2(tt_seq, triggers, sequence, n_iter)
    return tt_seq


def fetch_tones(folder):
    """
pour l'insatnt sans les mock a ajouter !!!!!
    """
    t_path = os.path.join(folder, "tones")
    tracking_pattern = "*tracking_0*.bin"
    mock_pattern = "*tracking_mock*.bin"
    pb_pattern = "*playback_*.bin"
    warmup_pattern = "*tail_*.bin"

    l_tracking = iterate_tones_folder(t_path, tracking_pattern)

    l_mock = iterate_tones_folder(t_path, mock_pattern)

    l_pb = iterate_tones_folder(t_path, pb_pattern)

    l_warmup = iterate_tones_folder(t_path, warmup_pattern)

    return l_tracking, l_mock, l_pb, l_warmup

def nan_sum(x):
    return np.sum(np.isnan(x[1]))


def has_nan(x):
    return np.isnan(x[1])


# Files part.
def get_recording_length(folder):
    assert (os.path.exists(os.path.join(folder, "recording_length.bin"))), "No length file."
    with open(os.path.join(folder, "recording_length.bin"), "r") as f:
        length = int(f.read())
    return length


def save_recording_length(folder, length):
    with open(os.path.join(folder, "recording_length.bin"), "w") as f:
        f.write('{:03d}\n'.format(length))


def load_files(file_list):
    out = dict()
    file_list.sort()
    for i, elt in enumerate(file_list):
        out[i] = np.load(elt)
    return out


def load_analog_files(file_list):
    out = dict()
    for elt in file_list:
        if re.search("Tracking", elt):
            out["tracking"] = np.load(elt)
        elif re.search("Mock", elt):
            out["mock"] = np.load(elt)
        elif re.search("Playback", elt):
            out["playback"] = np.load(elt)
    return out


def load_digital_files(file_list):
    out = dict()
    for elt in file_list:
        if re.search("Basler", elt):
            out["basler"] = np.load(elt)
        elif re.search("Sounds", elt):
            out["sounds"] = np.load(elt)
    return out


def check_plot_folder_exists(directory):
    path = os.path.join(directory, "plot")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def check_already_extracted(folder):
    tt = False
    length = False

    if os.path.exists(os.path.join(folder, "tt.npz")):
        tt = True

    if os.path.exists(os.path.join(folder, "recording_length.bin")):
        length = True

    return tt, length


def extract_tt(folder):
    tt_file = os.path.join(folder, "tt.npz")
    assert(os.path.exists(tt_file)), "No tt file."
    seq = SequenceTT(folder=folder)  # Aïe ?
    return seq


def check_digital_triggers(folder):
    return check_files(folder, analog=False)


def check_analog_triggers(folder):
    return check_files(folder, analog=True)


def check_files(folder, analog=True):
    """

    """
    if analog:
        fn_pattern = os.path.join(folder, "trig_analog_chan*.npy")
    else:
        fn_pattern = os.path.join(folder, "trig_dig_chan*.npy")
    lf = list(glob.glob(fn_pattern))
    return len(lf), lf


def process_digital_file(folder):
    """
    Prend un dossier en entrée. Cherche les noms de fichiers digitaux et extrait les temps de triggers.
    """
    return process_file(folder, analog=False)


def process_analog_file(folder, compatibility=False):
    """
    Prend un dossier en entrée. Cherche les noms de fichiers analogiques et extrait les temps de triggers.
    """
    return process_file(folder, analog=True, compatibility=compatibility)


def process_file(folder, analog=True, compatibility=False):
    """
    Comment sortir proprement les triggers mock?
    Est-ce que c'est à passer dans une classe
    """
    out = dict()
    if analog:
        f = os.path.join(folder, "analog_in.npy")

        if compatibility:
            fn_pattern = os.path.join(folder, "trig_analog_chan{}.npy")
            func = ut.extract_analog_triggers_compat
        else:
            fn_pattern = os.path.join(folder, "trig_analog_chan_{}.npy")
            func = ut.extract_analog_triggers
    else:
        f = os.path.join(folder, "dig_in.npy")
        fn_pattern = os.path.join(folder, "trig_dig_chan_{}.npy")
        func = ut.extract_digital_triggers
    triggers = np.load(f)
    n_channel = triggers.shape[0]
    for i in range(n_channel):
        events = func(triggers[i])
        if analog:
            if i == ANALOG_TRIGGERS_MAPPING["Main"]:
                if not compatibility:
                    np.save(fn_pattern.format("Tracking"), events[0])
                    np.save(fn_pattern.format("Mock"), events[1])
                    out["tracking"] = events[0]
                    out["mock"] = events[1]
                else:
                    np.save(fn_pattern.format("Tracking"), events)
                    out["tracking"] = events
            elif i == ANALOG_TRIGGERS_MAPPING["Playback"]:
                if not compatibility:
                    events = events[0]
                np.save(fn_pattern.format("Playback"), events)
                out["playback"] = events
        else:

            if i == DIGITAL_TRIGGERS_MAPPING["Basler"]:
                tag = "Basler"
                out["basler"] = events
            elif i == DIGITAL_TRIGGERS_MAPPING["Sounds"]:
                out["sounds"] = events
                tag = "Sounds"
            else:
                tag = i
            np.save(fn_pattern.format(tag), events)
    return out


def triggers_tones_inspection_2(tt_seq, triggers, sequence, n_iter):
    pb_triggers = triggers["playback"]
    tr_triggers = triggers["tracking"]
    mck_triggers = triggers["mock"]
    digital_triggers = triggers["dig"]
    pb_duration = sequence.get_duration_for("playback")

    tr_done = False
    pb_done = False
    mck_done = False

    s_pb = sequence.get_n_tones_for("playback")

    s_mck = sequence.get_n_tones_for("mock")

    s_tr = sum([sequence.get_n_tones_for(elt) for elt in ["tracking", "warmdown", "warmup"]])

    if s_tr == len(tr_triggers):
        tt_seq = if_complete_2(tr_triggers, sequence, tt_seq, type_of=None)
        tr_done = True

    if s_pb == len(pb_triggers):
        tt_seq = if_complete_2(pb_triggers, sequence, tt_seq, type_of="playback")
        pb_done = True

    if s_mck == len(mck_triggers):
        tt_seq = if_complete_2(mck_triggers, sequence, tt_seq, type_of="mock")
        mck_done = True

    where_to_withdraw = list()
    where_to_append = dict()
    if tr_done and pb_done and mck_done:
        return tt_seq, tr_done, pb_done, mck_done

    if not pb_done:
        # assembler les .bin tones
        # regarder quels triggers ne sont pas représentés dans les triggers analogiques et dans les triggers digitaux.

        p = deepcopy(pb_triggers)
        d = deepcopy(digital_triggers)
        d_corr_0 = synchronize_step(d, tr_triggers)
        d = d[np.isnan(d_corr_0[1])]
        a = list()  # triggers analogiques
        b = list()  # triggers digitaux
        for i in range(n_iter):
            start = p[0]
            end = p[0] + sequence.get_duration_for("playback")
            idx_p = np.less_equal(p, end)
            idx_d = np.logical_and(d >= start - 10000, d <= end)

            b.append(d[idx_d])
            a.append(p[idx_p])
            p = p[~idx_p]
        # 1) Assembler
        for i in range(n_iter):
            xp = sequence.get_xp_number("playback", i)
            _tones = xp.tones
            # _tones = np.vstack((np.full_like(_tones, i), _tones))
            # concat_pb_tones.append(_tones)

            if len(_tones) != len(a[i]):
                d_corr = synchronize_step(b[i], a[i], begin=False)
                if has_nan(d_corr[1]):
                    _tones = _tones[~np.isnan(d_corr[1])]
                else:
                    tr = sequence.get_xp_number("tracking", i).tones
                    if len(tr) == len(a[i]):
                        _tones = tr

            tt_seq.add(Pair(_tones, a[i], xp.type, number=xp.n, order=xp.order))

        pb_done = True

    if not tr_done:
        cp_tr_t = deepcopy(tr_triggers)
        # attraper les premiers triggers pour le warmup.
        xp = sequence.get_xp_number("warmup", 0)
        t0 = tr_triggers[0]
        idx = np.equal(xp.tones, 0)

        # 1) on sors le warmup et le warmdown.
        if sum(idx) > 0:
            xp.tones = xp.tones[~idx]
        wp_duration = sequence.get_duration_for("warmup")
        idx = np.less_equal(cp_tr_t, pb_triggers[0] - pb_duration)
        warmup_triggers = cp_tr_t[idx]
        if len(warmup_triggers) != len(xp.tones):
            if len(warmup_triggers) > len(xp.tones):
                d = len(warmup_triggers) - len(xp.tones)
                warmup_triggers = warmup_triggers[d:]
            else:
                # d = len(xp.tones) - len(warmup_triggers)
                xp.tones = xp.tones[:-1]

        tt_seq.add(Pair(xp.tones, warmup_triggers, xp.type, number=xp.n, order=xp.order))
        cp_tr_t = cp_tr_t[~idx]
        idx = np.greater_equal(cp_tr_t, pb_triggers[-1])
        xp = sequence.get_xp_number("warmdown", 0)
        wd_triggers = cp_tr_t[idx]
        tt_seq.add(Pair(xp.tones, wd_triggers, xp.type, number=xp.n, order=xp.order))
        cp_tr_t = cp_tr_t[~idx]

        # 2) on fait l'extraction des blocks tracking.
        if pb_done:
            l_tr_blocks = list()
            for i in range(n_iter):
                tt_after = tt_seq.get_xp_number("playback", i).triggers[0]
                if i > 0:
                    tt_before = tt_seq.get_xp_number("playback", i - 1).triggers[-1]
                    idx = np.logical_and(tt_before <= cp_tr_t, tt_after >= cp_tr_t)
                elif i == 0:
                    idx = np.logical_and(t0 + wp_duration <= cp_tr_t, tt_after >= cp_tr_t)
                l_tr_blocks.append(cp_tr_t[idx])
                cp_tr_t = cp_tr_t[~idx]

            # retirer indice 0 dans les triggers du tracking, si nécessaire.
            for i, elt in enumerate(l_tr_blocks):
                xp = sequence.get_xp_number("tracking", i)
                print(len(xp.tones), len(elt))
                if i == 0 and len(xp.tones) > len(elt):  # cas rare où le ton du warmup se trouve dans le tracking.
                    xp.tones = xp.tones[1:]

                if len(xp.tones) + 1 == len(elt):  # cas commun où le ton du mock se trouve dans le tracking.
                    if i != 0:
                        where_to_withdraw.append(i - 1)
                        mck_xp = sequence.get_xp_number("mock", i - 1)
                        xp.tones = np.hstack((mck_xp.tones[-1], xp.tones))

                if len(elt) + 1 == len(xp.tones):
                    if i != 0:
                        where_to_append[i] = xp.tones[-1]
                        xp.tones = xp.tones[:-1]

                tt_seq.add(Pair(xp.tones, l_tr_blocks[i], xp.type, number=xp.n, order=xp.order))

            tr_done = True

    if not mck_done:
        mck_duration = sequence.get_duration_for("mock")
        l_mck_blocks = list()
        cp_mck_t = deepcopy(mck_triggers)
        if tr_done:
            for i in range(n_iter):
                if len(sequence.get_xp_number("mock", i).tones) == 0:
                    l_mck_blocks.append([])
                    continue
                if i == n_iter - 1:
                    start = tt_seq.get_xp_number("tracking", i).triggers[-1]
                    stop = tt_seq.get_xp_number("warmdown", 0).triggers[0]
                else:
                    start = tt_seq.get_xp_number("tracking", i).triggers[-1]
                    stop = tt_seq.get_xp_number("tracking", i + 1).triggers[0]
                idx = np.logical_and(start <= cp_mck_t, stop >= cp_mck_t)
                l_mck_blocks.append(cp_mck_t[idx])
                cp_mck_t = cp_mck_t[~idx]
        else:
            for i in range(n_iter):
                end = cp_mck_t[0] + mck_duration
                if len(sequence.get_xp_number("mock", i).tones) == 0:
                    l_mck_blocks.append([])
                    continue
                idx = np.less_equal(cp_mck_t, end)
                l_mck_blocks.append(cp_mck_t[idx])
                cp_mck_t = cp_mck_t[~idx]

        for i, elt in enumerate(l_mck_blocks):

            xp = sequence.get_xp_number("mock", i)

            if i in where_to_withdraw:
                tt_seq.add(Pair(xp.tones[:-1], l_mck_blocks[i], xp.type, number=xp.n, order=xp.order))

            elif i in where_to_append.keys():
                tt_seq.add(Pair(xp.tones, l_mck_blocks[i], xp.type, xp.n, xp.order))

            else:
                if len(xp.tones) == len(l_mck_blocks[i]):
                    tt_seq.add(Pair(xp.tones, l_mck_blocks[i], xp.type, number=xp.n, order=xp.order))
                else:
                    tt_seq.add(Pair(xp.tones[:-1], l_mck_blocks[i], xp.type, number=xp.n, order=xp.order))

    return tt_seq


def create_data_folder(folder):
    """
    Créée le dossier data dans le dossier de l'animal.
    """
    data_folder = os.path.join(folder, "data")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    return data_folder


def check_data_folder_exists(folder):
    data_folder = os.path.join(folder, "data")
    if os.path.exists(data_folder):
        return True
    else:
        return False


def load_data_file_if_exists(folder, file):
    """
    Regarde si le fichier existe, le charge si c'est le cas.
    """
    check_data_folder_exists(folder)
    fn = os.path.join(folder, file)
    if os.path.exists(fn):
        return np.load(fn)
    else:
        return False


def create_data_file(data_folder, trial_type, cluster=None, trial_num=None, block=None, sub_block=None):
    fn = f"psth_{trial_type}"
    if cluster is not None:
        fn += f"_cl{cluster}"
    if trial_num is not None:
        fn += f"it{trial_num}"
    if block is not None:
        fn += f"blk{block}"
    if sub_block is not None:
        fn += f"sb{sub_block}"
    fn = os.path.join(data_folder, fn + ".npy")
    return fn

def save_psth_file(x, data_filename):
    # todo : Créer un authorised_type_name de type liste.
    np.save(data_filename, x)
    pass


def load_data_file(data_file):
    return np.load(data_file)

# todo : à supprimer.
def merge_pattern(dict_singleton, pattern):
    trigs_array = np.empty(0, dtype=int)
    tones_array = np.empty(0, dtype=np.double)
    # cas particulier pour le playback : enlever la 1ʳᵉ fréquence
    trim = False
    if pattern == "pb_[0-9]":
        trim = True
    for k in dict_singleton.keys():

        if re.search(pattern, k):
            if not trim:
                tones_array = np.hstack((tones_array, dict_singleton[k][0]))
                trigs_array = np.hstack((trigs_array, dict_singleton[k][1]))

            else:
                tones_array = np.hstack((tones_array, dict_singleton[k][0][1:]))
                trigs_array = np.hstack((trigs_array, dict_singleton[k][1][1:]))
    return tones_array, trigs_array


def merge_pattern_2(dict_singleton, pattern):
    array = list()
    # cas particulier pour le playback: enlever la 1ère fréquence
    for k in dict_singleton.keys():
        if re.search(pattern, k):
            array.append([dict_singleton[k][0], dict_singleton[k][1]])

    return array


def merge_pattern_3(dict_singleton, pattern):
    trigs_array = np.empty(0, dtype=int)
    tones_array = np.empty(0, dtype=np.double)
    # cas particulier pour le playback: enlever la 1ère fréquence
    for k in dict_singleton.keys():

        if re.search(pattern, k):
            tones_array = np.hstack((tones_array, dict_singleton[k][0]))
            trigs_array = np.hstack((trigs_array, dict_singleton[k][1]))

    return tones_array, trigs_array


def merge_pattern_4(dict_singleton, pattern):
    array = dict()
    k_list = list()
    # cas particulier pour le playback: enlever la 1ère fréquence
    for i, k in enumerate(list(dict_singleton.keys())):
        if re.search(pattern, k):
            k_list.append(k)
    k_list.sort()
    for i, elt in enumerate(k_list):
        array[i] = [dict_singleton[elt][0], dict_singleton[elt][1]]

    return array


def get_data3(folder, triggers):
    # todo: regarder les triggers digitaux. Regarder la concordance du canal digital avec le canal analogique.
    l_tracking, l_mock, l_pb, l_warmup = fetch_tones(folder)

    # ordonner les séquences.
    triggers_tracking = triggers["tracking"]  # élimination de l'artéfact de démarrage
    d_out = dict()
    warmup_pattern = "warmup_"
    trigs, tones_wu, counter = catch_triggers_and_tones(triggers_tracking, l_warmup[0], 0, maximum_duration=30)
    d_out[warmup_pattern + str(0)] = [tones_wu, trigs]
    trigs, tones_wu, _ = catch_triggers_and_tones(triggers_tracking, l_warmup[1],
                                                  counter=len(triggers_tracking) - len(l_warmup[1]),
                                                  maximum_duration=45)
    d_out[warmup_pattern + str(1)] = [tones_wu, trigs]
    assert (len(l_pb) == len(l_tracking) == len(l_mock))
    n_iter = len(l_pb)
    mock_out_pattern = "mock_"
    tracking_out_pattern = "tracking_"
    pb_out_pattern = "pb_"

    for i in range(n_iter):
        # assert(np.array_equal(l_pb[i][1:], l_tracking[i])), "array not equal."
        tr_pb, tones_pb, counter_pb = catch_triggers_and_tones(triggers["playback"], l_pb[i], counter_pb,
                                                               maximum_duration=5)
        d_out[pb_out_pattern + str(i)] = [tones_pb, tr_pb]
        tr_tr, tones_tr, counter = catch_triggers_and_tones(triggers_tracking, l_tracking[i], counter,
                                                            maximum_duration=5)
        d_out[tracking_out_pattern + str(i)] = [tones_tr, tr_tr]
        tr_mck, tones_mck, counter = catch_triggers_and_tones(triggers_tracking, l_tracking[i], counter,
                                                              maximum_duration=5)
        d_out[mock_out_pattern + str(i)] = [tones_mck, tr_mck]

    kwargs = {key: d_out[key] for key in d_out.keys()}
    np.savez(os.path.join(folder, "tt.npz"), **kwargs)
    return d_out


def get_data_2(folder, triggers):
    t_path = os.path.join(folder, "tones")
    tracking_pattern = "tracking_0*.bin"
    mock_pattern = "tracking_mock*.bin"
    pb_pattern = "playback_*.bin"
    warmup_pattern = "warmup_*.bin"

    l_tracking = iterate_tones_folder(t_path, tracking_pattern)

    l_mock = iterate_tones_folder(t_path, mock_pattern)

    l_pb = iterate_tones_folder(t_path, pb_pattern)

    l_warmup = iterate_tones_folder(t_path, warmup_pattern)

    length_channel_0 = 0
    for elt in l_warmup:
        length_channel_0 += len(elt)

    for elt in l_tracking:
        length_channel_0 += len(elt)

    for elt in l_mock:
        length_channel_0 += len(elt)

    # organiser le tout.

    sequence_chan_0 = np.empty(0)
    sequence_chan_0 = np.hstack((sequence_chan_0, l_warmup[0]))
    for i in range(len(l_mock)):
        sequence_chan_0 = np.hstack((sequence_chan_0, l_tracking[i]))
        sequence_chan_0 = np.hstack((sequence_chan_0, l_mock[i]))

    if len(l_warmup) > 1:
        sequence_chan_0 = np.hstack((sequence_chan_0, l_warmup[1]))

    sequence_pb = np.empty(0)
    for elt in l_pb:
        sequence_pb = np.hstack((sequence_pb, elt))

    # ordonner les séquences.
    triggers_tracking = triggers["tracking"]  # élimination de l'artéfact de démarrage
    d_out = dict()
    warmup_pattern = "warmup_"
    for i, elt in enumerate(l_warmup):
        if i == 0:
            d_out[warmup_pattern + str(i)] = [l_warmup[i], triggers_tracking[:len(l_warmup[i])]]
        else:
            d_out[warmup_pattern + str(i)] = [l_warmup[i], triggers_tracking[-len(l_warmup[i]):]]
    pb_out_pattern = "pb_"

    counter = 0
    for i, elt in enumerate(l_pb):

        t = elt[:-1]
        print(len(t) - len(triggers["playback"][counter:counter + len(t)]))
        foo = triggers["playback"][counter:counter + len(t)]
        ok = sanity_check(foo, 5)
        if not ok:
            print(f"Fishy PB{i}")
        if i != 5:
            d_out[pb_out_pattern + str(i)] = [t, triggers["playback"][counter:counter + len(t)]]
        else:
            d_out[pb_out_pattern + str(i)] = [t, triggers["playback"][counter:counter + len(t)]]
        counter += len(t)

    mock_out_pattern = "mock_"
    tracking_out_pattern = "tracking_"
    counter = len(l_warmup[0])
    for i in range(len(l_mock)):
        d_out[tracking_out_pattern + str(i)] = [l_tracking[i], triggers_tracking[counter:counter + len(l_tracking[i])]]
        ok = sanity_check(triggers_tracking[counter:counter + len(l_tracking[i])])
        if not ok:
            print(f"Fishy TR{i}")
        counter += len(l_tracking[i])
        #d_out[mock_out_pattern + str(i)] = [l_mock[i], triggers_tracking, c, 0, duration_warmdown, tones=l_warmup[1]]


    kwargs = {key: d_out[key] for key in d_out.keys()}
    np.savez(os.path.join(folder, "tt.npz"), **kwargs)
    return d_out


def assign_triggers(pb_channel, tr_channel, digital_channel, sequence):
    d_out = dict()
    sequence["playback"] = {"t": 5 * 60 * 30000, "n": 6}
    sequence["tracking"] = {"t": 5 * 60 * 30000, "n": 6}
    sequence["warmup"] = {"t": 30 * 60 * 30000, "n": 1}
    sequence["warmdown"] = {"t": 45 * 60 * 30000, "n": 1}

    n, t = sequence["warmup"]["n"], sequence["warmup"]["t"]
    idx = np.logical_and(tr_channel <= tr_channel[0], tr_channel >= tr_channel + t)
    to_out = tr_channel[idx]
    resync_analog_digital(to_out, digital_triggers=digital_channel, begin=True)
    tr_channel = tr_channel[~idx]

    for key in sequence.keys():
        n, t = sequence[key]["n"], sequence[key]["t"]
        if key == "playback":
            pb_channel, d_out = extract_with_times(analog_triggers=pb_channel, digital_triggers=digital_channel,
                                                   n=n, t=t, d_out=d_out, pattern="pb_")
        elif key == "tracking":
            tr_channel, d_out = extract_with_times(analog_triggers=tr_channel, digital_triggers=digital_channel,
                                                   n=n, t=t, d_out=d_out, pattern="tr_")

    return d_out


def extract_with_times(analog_triggers, digital_triggers, n, t, d_out, pattern):
    """

    """
    if pattern == "warmup_":
        begin = True
    else:
        begin = False
    for i in range(n):
        idx = np.logical_and(analog_triggers <= analog_triggers[0], analog_triggers >= analog_triggers + t)
        to_out = analog_triggers[idx]
        resync_analog_digital(to_out, digital_triggers=digital_triggers, begin=begin)
        analog_triggers = analog_triggers[~idx]
        d_out[pattern + str(i)] = to_out
    return analog_triggers, d_out


def get_n_tones(folder):
    tonotopy_seq = np.empty(0)
    for file in glob.glob(os.path.join(folder, "tones_*.bin")):
        tonotopy_seq = np.hstack((tonotopy_seq, np.fromfile(file, dtype=np.double)))
    return len(tonotopy_seq)


def get_tracking(folder, trigs):
    n_tones = get_n_tones(folder)
    return __tracking(folder, trigs, n_tones=n_tones)


def __tracking(folder, trigs, n_tones=None):
    if n_tones:
        trigs = trigs[n_tones:-n_tones]
    sequence = np.empty(0)
    for file in glob.glob(os.path.join(folder, "tracking_*.bin")):
        sequence = np.hstack((sequence, np.fromfile(file, dtype=np.double)))
    f = np.unique(sequence)
    print(len(sequence))
    sequence = sequence[:-1]
    print(len(sequence))
    return f, sequence, trigs, "_tracking"


def resync(digital, analog, tones, max_d=0.005, fs=30e3, begin=False, mock=False):
    a = synchronize_step(analog, digital, max_d, fs, begin)
    d = synchronize_step(digital, analog, max_d, fs, begin)
    len_a = a.shape[1]
    len_d = d.shape[1]
    len_t = len(tones)
    nan_d = np.isnan(d[1])  # nan du canal digital => ce sont ceux à enlever.
    nan_a = np.isnan(a[1])  # nan du canal analogique => ce sont ceux à garder.
    if len_t == len_a:
        return [tones, analog]
    elif len_t != len_a:
        if np.sum(nan_a) > 0:
            # on ajoute les triggers digitaux manquant d'après les triggers analogiques
            to_add = a[0][nan_a]
            to_add = np.vstack([to_add, to_add])
            d = np.hstack([d, to_add])
            d = d[:, np.argsort(d[0])]  # pas de nan ici
        # ce sont les triggers digitaux en trop
        if np.sum(nan_d) > 0:
            if len_t == len_d:
                d = d[~nan_d]
                tones = tones[~nan_d]
            else:
                pass

    # autant de triggers analogiques que de tons?
    # autant de triggers digitaux que de triggers analogiques?
    # if not => regarder les nan dans le sortie de sync du canal digital?
    l_tones = list()
    l_dig = list()
    l_analog = list()
    return


def synchronize_step(x, y, max_d=0.005, fs=30e3, begin=False):
    l_sync = list()
    max_d = max_d * fs
    if begin:
        l_sync.append([x[0], y[0]])
        start = 1
    else:
        start = 0
    for i in range(start, len(x)):
        idx = np.logical_and(x[i] - max_d < y, x[i] + max_d > y)
        if np.sum(idx) == 0:
            conc = np.nan
        elif np.sum(idx) == 1:
            conc = y[idx][0]
        else:
            # todo: gérer ce cas là
            conc = y[idx][0]
            print("Fishy")
        l_sync.append([x[i], conc])
    l_sync = np.transpose(np.array(l_sync))
    return l_sync


def align(digital, analog, tones, max_d=0.005, fs=30e3, begin=False):
    lt, ld, la = len(tones), len(digital), len(analog)
    d = synchronize_step(digital, analog, max_d, fs, begin=begin)
    a = synchronize_step(analog, digital, max_d, fs, begin=begin)
    nan_a = np.isnan(a[1])
    sa = np.sum(nan_a)  # on peut garder ça, car si trigger analogique => son diffusé.
    nan_d = np.isnan(d[1])  # on doit les jeter.
    sd = np.sum(nan_d)
    if lt == la == ld and sd == sa == 0:  # cas idéal
        return TT(tones, analog)
    elif lt == la != ld:

        assert (sd or sa > 0), "Debug needed."
        pass
    elif ld == lt:
        assert (sd > 0), "Debug needed."

        pass
    else:
        if lt == ld:
            assert (sd > 0), "Debug needed."
            assert (ld - sd == la), "Debug needed."

    pass


def divide_triggers(analog_triggers, pb_triggers, digital_channel, sequence, d_out, n_iter, pb=False):
    """
    la séquence doit être dans l'ordre.
    """
    tt_seq, tr_done, pb_done = triggers_tones_inspection(analog_triggers, pb_triggers, digital_channel, sequence, n_iter)
    if tr_done and pb_done:
        return tt_seq
    a_tr_tot = synchronize_step(analog_triggers, digital_channel, begin=True)
    d_tr_tot = synchronize_step(digital_channel, analog_triggers, begin=True)
    a_pb_tot = synchronize_step(pb_triggers, digital_channel, begin=False)
    d_pb_tot = synchronize_step(digital_channel, pb_triggers, begin=False)
    # todo: pour gérer les inconsistances entre len(tones), utiliser les triggers du playback
    # todo: regarder la concordance des triggers digitaux avec les triggers analogiques.
    # tt_seq = SequenceTT()
    copy_pb_triggers = deepcopy(pb_triggers)
    pb_duration = sequence.get_duration_for("playback")

    pb_list = sequence.get_all_xp_for_type("playback")
    s_pb = 0
    for elt in pb_list:
        s_pb += len(elt.tones)

    total_tones_tr = list()
    # todo: mettre dans l'ordre.
    mck_list = sequence.get_all_xp_for_type("mock")
    s_mck = 0
    for elt in mck_list:
        s_mck += len(elt.tones)

    tr_list = sequence.get_all_xp_for_type("tracking")
    s_tr = 0
    for elt in tr_list:
        s_tr += len(elt.tones)
    warmdown = sequence.get_all_xp_for_type("warmdown")[0]
    warmup = sequence.get_all_xp_for_type("warmup")[0]

    s_wd = len(warmdown.tones)
    s_wp = len(warmup.tones)

    s_tot = s_wp + s_wd + s_tr + s_mck
    if s_tot == len(analog_triggers):
        pass

    if s_pb == len(pb_triggers):
        pass

    last_pb_dig_trig = 0
    for i in range(n_iter):  # construire l'objet séquence. qui contient des objet XPSingleton
        copy_pb_triggers = deepcopy(pb_triggers)
        copy_pb_triggers -= copy_pb_triggers[0]
        pb_ref = pb_triggers[0]
        single_mck = mck_list[i]
        single_pb = pb_list[i]
        idx = np.logical_and(copy_pb_triggers >= 0, copy_pb_triggers <= pb_duration)
        idx_dig = np.logical_and(digital_channel >= pb_ref - 100, digital_channel <= pb_ref + pb_duration)

        mock_space = [digital_channel[idx_dig][0], digital_channel[idx_dig][-1]]
        idx_mck = np.logical_and(analog_triggers >= mock_space[0], analog_triggers <= mock_space[1])

        to_out_mck = analog_triggers[idx_mck]  # faire qqch avec ça.
        to_out_pb = pb_triggers[idx]
        to_out_pb_dig = digital_channel[idx_dig]
        last_pb_dig_trig = to_out_pb_dig[-1]

        # virer là où on a un NaN pour le trigger analogique
        tt_pb = align(to_out_pb_dig, to_out_pb, single_pb.tones)
        # tt_seq.add(Pair(np.array([]), np.array([]), single_mck.type, number=single_mck.n, order=single_mck.order))
        # tt_seq.add(Pair(tt_pb.tones, tt_pb.triggers, single_pb.type, number=single_pb.n, order=single_pb.order))
        copy_pb_triggers = copy_pb_triggers[~idx]
        pb_triggers = pb_triggers[~idx]
        digital_channel = digital_channel[~idx_dig]
        analog_triggers = analog_triggers[~idx_mck]
    d = synchronize_step(digital_channel, analog_triggers, begin=True)
    a = synchronize_step(analog_triggers, digital_channel, begin=True)

    # chercher d'abord warmup / warmdown
    warmup = sequence.get_all_xp_for_type("warmup")
    warmup_duration = sequence.get_duration_for("warmup")
    reference_trigger = analog_triggers[0]
    idx_dig = np.logical_and(digital_channel >= reference_trigger, digital_channel <= reference_trigger + warmup_duration)
    idx = np.logical_and(analog_triggers >= reference_trigger, analog_triggers <= reference_trigger + warmup.t)
    to_out = analog_triggers[idx]
    to_out_dig = digital_channel[idx_dig]
    if len(warmup.tones) == len(to_out):
        d_out[warmup.pattern + str(warmup.n)] = [warmup.tones, to_out]
    elif len(warmup.tones) == len(to_out_dig):
        # on doit éliminer la fréquence en trop
        d_out[warmup.pattern + str(warmup.n)] = eliminate_intruder_tone(to_out, to_out_dig, warmup.tones, True)
    analog_triggers = analog_triggers[~idx]
    digital_channel = digital_channel[~idx_dig]
    reference_trigger += warmup.t
    first_val = np.where(digital_channel > last_pb_dig_trig)[0][0]
    idx_dig = np.greater(digital_channel, first_val)
    idx = np.greater(analog_triggers, first_val)
    to_out = analog_triggers[idx]
    to_out_dig = digital_channel[idx_dig]
    tt = align(to_out_dig, to_out, warmdown.tones)
    tt_seq.add(Pair(tt.tones, tt.triggers, warmdown.type, order=warmdown.order))


    tr_duration = sequence.get_duration_for("tracking")
    for i in range(n_iter):
        single_tr = tr_list[i]
        copy_analog_triggers = deepcopy(analog_triggers)
        copy_analog_triggers -= copy_analog_triggers[0]
        copy_digital_triggers = deepcopy(digital_channel)
        copy_digital_triggers -= copy_digital_triggers[0]

        idx = np.where(copy_analog_triggers >= 0, copy_analog_triggers <= tr_duration)
        idx_dig = np.where(copy_digital_triggers >= 0, copy_digital_triggers <= tr_duration)
        to_out = analog_triggers[idx]
        to_out_dig = digital_channel[idx_dig]
        tt = align(to_out, to_out_dig, single_tr.tones)
        tt_seq.add(Pair(tt.tones, tt.triggers, single_tr.type, number=single_tr.n, order=single_tr.order))
        digital_channel = analog_triggers[~idx]
        analog_triggers = digital_channel = digital_channel[~idx_dig]
    # resynchroniser
    for elt in sequence[1:-1]:
        if elt.type == "warmup":
            begin = True

        else:
            begin = False
        if pb:
            reference_trigger = analog_triggers[0]
        b, e = reference_trigger, reference_trigger + elt.t
        if not pb:
            reference_trigger += elt.t
        idx_dig = np.logical_and(digital_channel >= b, digital_channel <= e)
        to_out_dig = digital_channel[idx_dig]

        if elt.type != "mock":
            idx = np.logical_and(analog_triggers >= b, analog_triggers <= e)
            to_out = analog_triggers[idx]
            print(len(elt.tones))
            print(len(to_out))
            print(len(to_out_dig))
            # tones_triggers = resync(to_out_dig, to_out, elt.tones, begin=begin)
        else:
            # si mock on utilise les trigs digitaux
            b, e = to_out_dig[0], to_out_dig[0] + elt.t

            idx = np.logical_and(analog_triggers >= b, analog_triggers <= e)
            to_out = analog_triggers[idx]
            print("MOCK")
            print(len(elt.tones))
            print(len(to_out))
            # tones_triggers = [elt.tones, to_out]

        if len(to_out) == len(elt.tones):
            tones_triggers = [elt.tones, to_out]
        elif len(to_out_dig) == len(elt.tones):
            tones_triggers = eliminate_intruder_tone(to_out, to_out_dig, elt.tones)
        else:
            tones_triggers = [0, 0]
        d_out[elt.pattern + str(elt.n)] = tones_triggers
        analog_triggers = analog_triggers[~idx]
        digital_channel = digital_channel[~idx_dig]
    return d_out


def merge_and_sync(d, x, y, max_d=0.005, fs=30e3, begin=False):
    l_sync = list()
    max_d = max_d * fs
    if begin:
        l_sync.append([d[0], x[0], 0])
        start = 1
    else:
        start = 0
    for i in range(start, len(d)):
        val = d[i]
        idx_x = np.logical_and(val - max_d < x, val + max_d > x)
        idx_y = np.logical_and(val - max_d < y, val + max_d > y)
        if np.sum(idx_y) == 0:
            if np.sum(idx_x) == 0:
                conc = np.nan
                belongs = -1
            elif np.sum(idx_x) == 1:
                conc = x[idx_x][0]
                belongs = 0

        elif np.sum(idx_y) == 1:
            conc = y[idx_y][0]
            belongs = 1
        else:
            # todo: gérer ce cas là
            conc = y[idx_y][0]
            belongs = -2
            print("Fishy")
        l_sync.append([val, conc, belongs])
    l_sync = np.transpose(np.array(l_sync))
    return l_sync


def triggers_tones_inspection(analog_tr, analog_pb, digital, sequence, n_iter):
    tt_seq = SequenceTT()

    tr_done = False
    pb_done = False

    s_pb = sequence.get_n_tones_for("playback")
 

    s_tr = sum([sequence.get_n_tones_for(elt) for elt in ["mock", "tracking", "warmdown", "warmup"]])

    if s_tr == len(analog_tr):
        tt_seq = if_complete(analog_tr, sequence, tt_seq)
        tr_done = True

    if s_pb == len(analog_pb):
        tt_seq = if_complete(analog_pb, sequence, tt_seq, pb=True)
        pb_done = True

    if tr_done and pb_done:
        return tt_seq, tr_done, pb_done

    elif not pb_done and tr_done:
        # d_tr_tot = synchronize_step(digital, analog_tr, begin=True)
        # d_pb_tot = synchronize_step(digital, analog_pb, begin=False)
        pb_duration = sequence.get_duration_for("PLAYBACK")
        mck_trigs = tt_seq.get_all_triggers_for_type("mock")
        keep = list()

        for elt in analog_tr:
            if elt not in mck_trigs:
                keep.append(elt)
        d_keep = synchronize_step(digital, np.array(keep), begin=True)
        d_keep = d_keep[0][np.isnan(d_keep[1])]
        total = len(analog_pb)
        n_to_discard = len(d_keep) - total  # problème ici
        # d_keep: ce sont les triggers digitaux qui n'ont pas de correspondance dans sur le canal tracking.
        # ils correspondent donc aux triggers du playback

        # Séparation des triggers de playback.

        count = 0
        for i in range(n_iter):
            copy_pb_triggers = deepcopy(analog_pb)
            copy_keep = deepcopy(d_keep)

            if i == 0:
                ref = d_keep[0]
            single_pb = sequence.get_xp_number("playback", i)
            tones = single_pb.tones
            copy_pb_triggers = copy_pb_triggers.astype(np.float64) - copy_keep[0]
            copy_keep -= copy_keep[0]
            idx = np.logical_and(copy_pb_triggers >= 0, copy_pb_triggers <= pb_duration)
            idx_dig = np.logical_and(copy_keep >= 0, copy_keep <= pb_duration)
            triggers = analog_pb[idx]
            d_triggers = d_keep[idx_dig]
            la, ld, lt = len(triggers), len(d_triggers), len(tones)
            count += la

            if la != lt:
                a = synchronize_step(copy_pb_triggers[idx], copy_keep[idx_dig])
                d = synchronize_step(copy_keep[idx_dig], copy_pb_triggers[idx])
                nan_a = nan_sum(a)
                nan_d = nan_sum(d)
                if lt == ld:
                    if nan_a == 0:
                        nan_idx = has_nan(d)
                        n_to_discard -= len(np.where(nan_idx == 1)[0])
                        tones = tones[~nan_idx]  # on garde les non nan.
                        triggers = analog_pb[idx]
                        # comme on s'assure que nombre de tones == nombre de trigs digitaux.

                    elif nan_a > 0:
                        nan_idx = has_nan(d)
                        n_to_discard -= len(np.where(nan_idx is True)[0])

                        triggers = analog_pb[idx]
                        tones = tones
                elif lt != la != ld:
                    # nan_idx = has_nan(d)
                    tones = tones[1:]
                    pass
                else:
                    triggers = analog_pb[idx]
            analog_pb = analog_pb[~idx]
            d_keep = d_keep[~idx_dig]
            tt_seq.add(build_pair_from_singleton(triggers, tones, single_pb))
        if count == total:
            pb_done = True

    elif not tr_done and pb_done:
        pb_trigs = tt_seq.get_all_triggers_for_type("playback")
        pb_duration = sequence.get_duration_for("playback")
        # a_tr_tot = synchronize_step(analog_tr, digital, begin=True)
        a_pb_tot = synchronize_step(pb_trigs, digital, begin=True)
        keep = list()
        pb_dig = list()

        for elt in digital:

            if elt in a_pb_tot[1]:
                pb_dig.append(elt)

            else:
                keep.append(elt)

        pb_dig = np.array(pb_dig, dtype=int)
        for_tracking = list()
        keep = np.array(keep, dtype=int)
        copy_dig = deepcopy(pb_dig)
        for i in range(n_iter):
            ref, end = copy_dig[0], copy_dig[0] + pb_duration
            to_discard = copy_dig
            to_discard -= to_discard[0]
            idx_discard = np.logical_and(to_discard >= 0, to_discard <= pb_duration)
            idx = np.logical_and(analog_tr >= ref, analog_tr <= end)
            triggers = analog_tr[idx]
            for_tracking.append(copy_dig[idx_discard])
            # p, idx_tr = mock_resync(i, sequence, copy_tr_triggers, d_triggers, pb_duration)

            single_mck = sequence.get_xp_number("mock", i)
            tones = single_mck
            lt, la = len(tones), len(triggers)
            if lt != la:
                tones = tones[:la]
            tt_seq.add(build_pair_from_singleton(triggers, tones, single_mck))
            analog_tr = analog_tr[~idx]
            copy_dig = copy_dig[~idx_discard]

        tr_seq = sequence.get_for_types(["warmup", "tracking", "warmdown"])
        copy_pb = deepcopy(pb_dig)
        for i, elt in enumerate(tr_seq):
            assert (elt.type == "warmup" and i == 0), "Not in order."
            duration = elt.t
            tones = elt.tones
            copy_tr_triggers = deepcopy(analog_tr)
            copy_dg_triggers = deepcopy(keep)

            if elt.type in ["warmup", "warmdown"]:
                if elt.type == "warmup":
                    bg = True
                else:
                    bg = False
                copy_dg_triggers = copy_dg_triggers - copy_tr_triggers[0]
                copy_tr_triggers -= copy_tr_triggers[0]
                idx = np.logical_and(copy_tr_triggers >= 0, copy_tr_triggers <= duration)
                idx_dig = np.logical_and(copy_dg_triggers >= 0, copy_dg_triggers <= duration)
                triggers = analog_tr[idx]
                d_triggers = digital[idx_dig]
                la, ld, lt = len(triggers), len(d_triggers), len(tones)

                if lt != la:
                    a = synchronize_step(copy_tr_triggers[idx], copy_dg_triggers[idx_dig], begin=bg)
                    d = synchronize_step(copy_dg_triggers[idx_dig], copy_tr_triggers[idx], begin=bg)

                    nan_a = nan_sum(a)
                    nan_d = nan_sum(d)

                    if nan_a > 0:
                        pass
                    else:
                        if lt == ld:
                            nan_idx = has_nan(d)
                            tones = tones[~nan_idx]

            else:
                end = for_tracking[elt.n][0]
                idx = np.less(analog_tr, end)
                idx_dig = np.less(digital, end)
                triggers = analog_tr[idx]
                d_triggers = digital[idx_dig]
                la, ld, lt = len(triggers), len(d_triggers), len(tones)
                if lt != la:
                    pass
                pass
            tt_seq.add(build_pair_from_singleton(triggers, tones, elt))

    else:
        n_tones_pb_discard = s_pb - len(analog_pb)
        n_tones_tr_discard = s_tr - len(analog_tr)
        # copy_pb = deepcopy(analog_pb)
        pb_duration = sequence.get_duration_for("playback")

        single_warmup = sequence.get_for_types("warmup")[0]
        wp_tones = single_warmup.tones
        wp_duration = single_warmup.t
        idx = np.logical_and(analog_tr >= analog_tr[0], analog_tr <= analog_tr[0] + wp_duration)
        idx_dig = np.logical_and(digital >= analog_tr[0] - 100, digital <= analog_tr[0] + wp_duration)
        triggers = analog_tr[idx]
        origin = analog_tr[0]

        if len(wp_tones) == len(triggers):
            tt_seq.add(build_pair_from_singleton(triggers, wp_tones, single_warmup))
            digital = digital[~idx_dig]
            analog_tr = analog_tr[~idx]
        ref = origin + wp_duration

        fishy = list()
        for i in range(n_iter):
            triple = sequence.get_all_number(i)
            single_tr = triple["tracking"]
            single_pb = triple["playback"]
            single_mck = triple["mock"]
            tr_tones = single_tr.tones
            idx = np.logical_and(analog_tr >= ref, analog_tr <= ref + single_tr.t)
            idx_dig = np.logical_and(digital >= ref, digital <= ref + single_tr.t)
            triggers = analog_tr[idx]
            d_triggers = digital[idx_dig]
            la, lt, ld = len(triggers), len(tr_tones), len(d_triggers)
            if lt == la:
                tt_seq.add(build_pair_from_singleton(triggers, tr_tones, single_tr))
                digital = digital[~idx_dig]
                analog_tr = analog_tr[~idx]
            else:
                a = synchronize_step(triggers, d_triggers)
                d = synchronize_step(d_triggers, triggers)
                if la == ld and la > lt:
                    pb_tones = single_pb.tones
                    if len(pb_tones) > lt and len(pb_tones) == la:
                        tt_seq.add(build_pair_from_singleton(triggers, pb_tones, single_tr))

                digital = digital[~idx_dig]
                analog_tr = analog_tr[~idx]
            ref += single_tr.t

            is_fishy = False
            pb_tones = single_pb.tones

            idx = np.logical_and(analog_pb >= ref, analog_pb <= ref + pb_duration)
            idx_dig = np.logical_and(digital >= ref, digital <= ref + pb_duration)

            triggers, d_triggers = analog_pb[idx], digital[idx_dig]

            la, ld, lt = len(triggers), len(d_triggers), len(pb_tones)
            if lt != la:
                a = synchronize_step(triggers, d_triggers)
                d = synchronize_step(d_triggers, triggers)
                nan_a = nan_sum(a)
                nan_d = nan_sum(d)
                if nan_a > 0:
                    # regarder si premier trigger a une correspondance.
                    pass
                else:  # nan_a == 0
                    if lt == ld:
                        nan_idx = has_nan(d)
                        pb_tones = pb_tones[~nan_idx]
                    else:
                        is_fishy = True
                        fishy.append(["playback", i])
                        # trt = sequence.get_xp_number("tracking", i)
            if not is_fishy:
                tt_seq.add(build_pair_from_singleton(triggers, pb_tones, single_pb))

            # MOCK
            p, idx_tr = mock_resync(single_mck, analog_tr, ref)
            tt_seq.add(p)
            analog_pb = analog_pb[~idx]
            analog_tr = analog_tr[~idx_tr]
            digital = digital[~idx_dig]
            ref += single_pb.t

        tr_seq = sequence.get_for_types("warmdown")
        for i, elt in enumerate(tr_seq):
            duration = elt.t
            tones = elt.tones

            idx = np.logical_and(analog_tr >= ref, analog_tr <= ref + duration)
            idx_dig = np.logical_and(digital >= ref, digital <= ref + duration)

            triggers = analog_tr[idx]
            d_triggers = digital[idx_dig]

            la, ld, lt = len(triggers), len(d_triggers), len(tones)
            if lt != la:
                a = synchronize_step(triggers, d_triggers)
                d = synchronize_step(triggers, d_triggers)

                nan_a = nan_sum(a)
                nan_d = nan_sum(d)

                if nan_d > 0:
                    if lt == ld:
                        if nan_a == 0:
                            nan_idx = has_nan(d)
                            n_tones_tr_discard -= len(np.where(nan_idx == 1)[0])
                            tones = tones[~nan_idx]
                        elif nan_a > 0:
                            idx_nan_a = np.where(has_nan(a) == 1)[0]
                            idx_nan_d = np.where(has_nan(d) == 1)[0]
                            if np.array_equal(idx_nan_a, idx_nan_d) and la == ld:
                                tones = tones[:la]
                    else:
                        tones = tones[:la]
            tt_seq.add(build_pair_from_singleton(triggers, tones, elt))
            analog_tr = analog_tr[~idx]
            digital = digital[~idx_dig]

    return tt_seq, True, True


def mock_resync(singleton, analog, t0):
    tones = singleton.tones
    idx = np.logical_and(analog >= t0, analog <= t0 + singleton.t)
    triggers = analog[idx]
    lt, la = len(tones), len(triggers)
    if lt != la:
        tones = tones[:la]
    return build_pair_from_singleton(triggers, tones, singleton), idx


def resync_digital_analog(analog_triggers, digital_triggers, max_d=0.005, fs=30e3, begin=False):
    pass


def eliminate_intruder_tone(analog, digital, tones, max_d=0.005, fs=30e3, begin=False):
    # a = synchronize_step(analog, digital, max_d, fs, False)
    d = synchronize_step(digital, analog, max_d, fs, begin)
    nan_d = np.isnan(d[1])
    tones = tones[~nan_d]
    return [tones, analog]


def get_data_4(triggers, folder):
    """
    Fait intervenir le canal digital. On regarde dans un intervalle de temps autour du trig digital.
    On doit trier d'abord les tones dans chaque catégorie. ex -> warmup avec warmup, pb avec pb etc.
    Quand y a un trigger analogique mais de digital => Pas de problème en soit
    Mais quand digital est là sans analogique: on vire.
    """
    pb_triggers = triggers["ANALOG"]["PLAYBACK"]
    tr_triggers = triggers["ANALOG"]["MAIN"]
    digital_triggers = triggers["DIGITAL"]["BASLER"]
    d_out = dict()
    l_tracking, l_mock, l_pb, l_warmup = fetch_tones(folder)

    print('longueur = ', l_tracking, l_pb)
    
    
    n_iter = len(l_pb)
    print('n_iter = ', n_iter)
    duration_tr = 5
    duration_warmup = 5
    duration_warmdown = 5

    # todo: prendre ces informations dans le json.
    c = 0
    sequence = Sequence()
    #sequence.add(XPSingleton("TrackingTail", c, 0, duration_warmup, tones=l_warmup[0]))
    #c += 1
    for i in range(n_iter):
        sequence.add(XPSingleton("tracking", c, i, duration_tr, tones=l_tracking[i]))
        c += 1
        #sequence.add(XPSingleton("mock", c, i, duration_tr, tones=l_mock[i]))
        #c += 1
        sequence.add(XPSingleton("playback", c, i, duration_tr, tones=l_pb[i]))
        c += 1

    #sequence.add(XPSingleton("TrackingTail", c, 0, duration_warmdown, tones=l_warmup[1]))

    d_out = divide_triggers(tr_triggers, pb_triggers, digital_triggers, sequence, d_out, n_iter=n_iter)
    d_out.set_n_iter(n_iter)
    return d_out


def resync_tracking_playback(dict_singleton):
    # 1 compter si nombre d'exp playback == tracking
    l_pb = list()
    l_tr = list()
    for k in dict_singleton.keys():

        if re.search("pb_[0-9]", k):
            l_pb.append(k)
        elif re.search("tracking_[0-9]", k):
            l_tr.append(k)

    l_pb.sort()
    l_tr.sort()
    for m in range(len(l_pb)):
        tr_m = dict_singleton[l_tr[m]]
        pb_m = dict_singleton[l_pb[m]]
    return


def if_complete(analog, sequence, tt, pb=False):
    xp_list = sequence.get_in_order(pb)
    for elt in xp_list:
        t = elt.tones
        triggers, analog = analog[:len(t)], analog[len(t):]
        tt.add(Pair(t, triggers, elt.type, number=elt.n, order=elt.order))
    return tt


def resync_analog_digital(analog_triggers, digital_triggers, max_d=0.005, fs=30e3, begin=False):
    l_tr_clean = list()
    max_d = max_d * fs
    #
    if begin:
        l_tr_clean.append([analog_triggers[0], digital_triggers[0]])
        start = 1
    else:
        start = 0
    for i in range(start, len(analog_triggers)):
        idx = np.logical_and(analog_triggers[i] - max_d < digital_triggers,
                             analog_triggers[i] + max_d > digital_triggers)
        l_tr_clean.append([analog_triggers[i], digital_triggers[idx][0]])
    l_tr_clean = np.array(l_tr_clean)
    return l_tr_clean


def catch_triggers_and_tones(triggers, tones, counter, maximum_duration=5):
    t = triggers[counter:counter + len(tones)]
    ok = sanity_check(t)

    if not ok:
        t, tones = clean(t, tones, maximum_duration)

    counter += len(tones)
    return t, tones, counter


def sanity_check(triggers, maximum_duration=5, fs=30e3):
    delta = triggers[-1] - triggers[0]
    delta /= fs
    delta /= 60
    if delta > maximum_duration:
        warnings.warn("Something fishy in triggers...", UserWarning)
        return False
    else:
        return True


def clean(triggers, tones, maximum_duration=5):
    start = triggers[0]
    deltas = (triggers - start) / 30000 / 60
    idx = np.logical_and(deltas >= 0, deltas <= maximum_duration)
    new_tones = tones[idx]
    triggers = triggers[idx]
    return triggers, new_tones

