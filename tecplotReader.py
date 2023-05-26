#!/usr/bin/env python3
'''Python script to load Tecplot binary file or gzipped binary file'''
__author__ = "Han Luo"
__copyright__ = "Copyright 2023, Han Luo"
__license__ = "GPL"
__version__ = "4.2.0"
__maintainer__ = "Han Luo"
__email__ = "aGFuLmx1b0BnbWFpbC5jb20="

import numpy as np
import enum
import construct  # work with v2.10.67
import functools
import operator
import gzip
import logging
import sys
from construct import Int8ul as Byte
from construct import Int16sl as Short
from construct import Int32sl as Int
from construct import Float32l as Float
from construct import Float64l as Double
from construct import Bit
from construct import Struct, Array, Const, this, GreedyRange, If, Index, Computed, Default, Rebuild, Check, len_, Tell, Peek
from construct import IfThenElse, Error, StopIf
from construct.lib.containers import ListContainer
from io import SEEK_CUR


class TecplotString(construct.Adapter):
    """Tecplot style string"""

    def __init__(self, encoder=construct.Int32ul):
        self.encoder = encoder
        subcon = construct.NullTerminated(construct.GreedyBytes, term=self.encoder.build(0))
        super().__init__(subcon)
        self.nbyte = self.encoder.sizeof()

    def _decode(self, obj, context, path):
        return "".join([chr(self.encoder.parse(obj[i:])) for i in range(0, len(obj), self.nbyte)])

    def _encode(self, obj, context, path):
        if obj == "":
            return b""
        return b"".join([self.encoder.build(ord(i)) for i in obj])

    def _emitparse(self, code):
        raise NotImplementedError

    def _emitbuild(self, code):
        raise NotImplementedError
        # This is not a valid implementation. obj.encode() should be inserted into subcon
        # return f"({self.subcon._compilebuild(code)}).encode({repr(self.encoding)})"

    def _emitfulltype(self, ksy, bitwise):
        return dict(type="strz", encoding="ascii")


TecStr = TecplotString()


def squeeze_ijk(ijk, d_ijk=None):
    if d_ijk is None:
        d_ijk = ijk
    for i in range(len(ijk) - 1, -1, -1):
        if d_ijk[i] != 1:
            return ijk[:i + 1] if i > 0 else [ijk[0]]
    return ijk[:]


class TecplotMatrix(construct.Construct):
    """Matrix"""

    def __init__(self, subcon, cell_centered=False, ijk=None, discard=False):
        super().__init__()
        self.name = "TecplotMatrix"
        self.subcon = subcon
        self.dtype = None
        if not callable(subcon):
            self.dtype = np.dtype(subcon.fmtstr)
        self.discard = discard
        self.cell_centered = cell_centered
        self.ijk = ijk
        self.order = 'F'

    @staticmethod
    def ijk_to_fshape(ijk, cell_centered=False):
        if cell_centered:
            if ijk[2] > 1:
                return [ijk[0], ijk[1], ijk[2] - 1]
            elif ijk[1] > 1:
                return [ijk[0], ijk[1] - 1]
            else:
                return [ijk[0] - 1]
        return ijk[:]

    @staticmethod
    def ijk_to_mshape(ijk, cell_centered=False):
        if cell_centered:
            t = []
            for i in range(3):
                if ijk[i] > 1:
                    t.append(ijk[i])
                else:
                    return t
            return t
        return ijk[:]

    @staticmethod
    def mshape_to_ijk(mshape, cell_centered=False):
        t = [i for i in mshape]
        if len(t) > 3:
            raise ValueError("Dimension should be less than 3")
        if cell_centered:
            t = [i + 1 for i in t]
        for i in range(len(t), 3):
            t.append(1)
        return t

    @staticmethod
    def fdata_to_mdata(data: np.ndarray, ijk, cell_centered=False):
        if cell_centered:
            if ijk[2] > 1:
                return data[:-1, :-1]
            elif ijk[1] > 1:
                return data[:-1, :]
            else:
                return data
        return data

    @staticmethod
    def mdata_to_fdata(data: np.ndarray, cell_centered=False):
        if cell_centered:
            if data.ndim == 1:
                return data.copy()
            elif data.ndim == 2:
                return np.vstack((data, np.zeros(data.shape[1], dtype=data.dtype)))
            elif data.ndim == 3:
                data = np.append(
                    data,
                    np.zeros_like(data[1, :, :]).reshape((1, data.shape[1], data.shape[2])),
                    axis=0
                )
                data = np.append(
                    data,
                    np.zeros_like(data[:, 1, :]).reshape((data.shape[0], 1, data.shape[2])),
                    axis=1
                )
                return data
            else:
                raise ValueError("cell centered only accepts dim <= 3")
        return data.copy()

    def _parse(self, stream, context, path):
        subcon = self.subcon
        dtype = self.dtype
        cell_centered = self.cell_centered
        discard = self.discard
        ijk = self.ijk
        if 'subcon' in context:
            subcon = context.subcon
        if callable(subcon):
            subcon = construct.evaluate(subcon, context)
            dtype = np.dtype(subcon.fmtstr)
        if 'cell_centered' in context:
            cell_centered = context.cell_centered
        cell_centered = bool(construct.evaluate(cell_centered, context))
        if 'discard' in context:
            discard = context.discard
        discard = construct.evaluate(discard, context)

        ijk = construct.evaluate(ijk, context)
        if len(ijk) < 3:
            for i in range(len(ijk), 3):
                ijk.append(1)
        fshape = self.ijk_to_fshape(ijk, cell_centered)

        length = functools.reduce(operator.mul, fshape) * dtype.itemsize
        if length <= 0:
            raise construct.RangeError("invalid length")
        context.length = length
        context.offset = construct.stream_tell(stream, path)
        context.discard = discard
        if discard:
            construct.stream_seek(stream, length, SEEK_CUR, path)
            return None
        else:
            obj = np.frombuffer(
                construct.stream_read(stream, length, path),
                dtype=dtype,
            ).reshape(fshape, order=self.order)
            mdata = self.fdata_to_mdata(obj, ijk, cell_centered)
            if path[-7:] == "min_max":
                return np.reshape(mdata, (2, -1))
            npshape = squeeze_ijk(mdata.shape, fshape)
            return mdata.reshape(npshape)

    def _build(self, obj: np.ndarray, stream, context, path):
        subcon = self.subcon
        if callable(subcon):
            subcon = subcon(context)
        dtype = np.dtype(subcon.fmtstr)
        obj = np.array(obj).astype(dtype)
        cell_centered = self.cell_centered
        if callable(cell_centered):
            cell_centered = cell_centered(context)
        discard = self.discard
        if 'discard' in context:
            discard = context.discard
        if not isinstance(cell_centered, bool):
            raise ValueError("cell_centered should be a bool")
        mshape = list(obj.shape)
        ijk = self.mshape_to_ijk(mshape, cell_centered)
        fshape = self.ijk_to_fshape(ijk, cell_centered)
        length = functools.reduce(operator.mul, fshape) * dtype.itemsize
        context.length = length
        context.offset = construct.stream_tell(stream, path)
        context.discard = discard
        if discard:
            buf = b'\xff' * length
        else:
            buf = self.mdata_to_fdata(obj, cell_centered).tobytes(order='F')
        construct.stream_write(stream, buf, length, path)
        return buf

    def _sizeof(self, context, path):
        raise construct.SizeofError(path=path)

    def _emitfulltype(self, ksy, bitwise):
        return dict(type=self.subcon._compileprimitivetype(ksy, bitwise), repeat="eos")


class ZoneType(enum.IntEnum):
    ORDERED = 0
    FELINESEG = 1
    FETRIANGLE = 2
    FEQUADRILATERAL = 3
    FETETRAHEDRON = 4
    FEBRICK = 5
    FEPOLYGON = 6
    FEPOLYHEDRON = 7


class VarLoc(enum.IntEnum):
    Node = 0,
    CellCentered = 1


class VarType(enum.IntEnum):
    Float = 1,
    Double = 2,
    LongInt = 3,
    ShortInt = 4,
    Byte = 5,
    Bit = 6


VarTypeConstruct = [None, Float, Double, Int, Short, Byte, Bit]


def calculate_data_length(zone_type, var_loc, ijk, num_pts, num_elems):
    if zone_type == ZoneType.ORDERED:
        if var_loc == VarLoc.Node:
            return ijk[0] * ijk[1] * ijk[2]
        else:
            return ijk[0] * (1 if ijk[1] <= 1 else (ijk[1] if ijk[2] <= 1 else ijk[1] * ijk[2]))
    else:
        return num_pts if var_loc == VarLoc.Node else num_elems


def calculate_nvar_zone(has_passive_var, passive_var, has_shared_var, shared_var, nvar):
    return len([1 for i, j in zip(passive_var if has_passive_var else [
        0] * nvar, shared_var if has_shared_var else [-1] * nvar) if i == 0 and j == -1])


def find_var_zone(has_passive_var, passive_var, has_shared_var, shared_var, nvar):
    return [i for i, (j, k) in enumerate(zip(passive_var if has_passive_var else [
        0] * nvar, shared_var if has_shared_var else [-1] * nvar))
        if j == 0 and k == -1]


def aux_parse(x, lst, ctx):
    if x == '':
        lst.pop()
        return True
    return False


# ===================== Tecplot Sections ======================

TecHeader = Struct(
    Const(b"#!TDV112"),  # Magic number
    Const(Int.build(1)),  # Integer value of 1
    Const(Int.build(0)),  # FileType: 0 = FULL
    "title" / Default(TecStr, ""),  # File Title
    "nvar" / Rebuild(Int, len_(this.variables)),  # Number of variables
    "variables" / Array(this.nvar, TecStr),  # Variable names
)

TecDatasetAux = Struct(
    Const(Float.build(799.0)),
    "name" / TecStr,  # Variable names
    Const(Int.build(0)),
    "value" / TecStr,
)

TecGeom = Struct(
    Const(Float.build(399.0)),
    "igeom" / Index,
    "position_coord_sys" / Default(Int, 0),
    "scope" / Default(Int, 0),
    "draw_order" / Default(Int, 0),
    "x0" / Default(Double, 0.0),
    "y0" / Default(Double, 0.0),
    "z0" / Default(Double, 0.0),
    "zone" / Default(Int, 0),
    "color" / Default(Int, -1),
    "fill_color" / Default(Int, -1),
    "is_filled" / Default(Int, 0),
    "type" / Default(Int, 0),
    "line_pattern" / Default(Int, 0),
    "pattern_length" / Default(Double, 0),  # GUI's value divided by 100
    "line_thickness" / Default(Double, 0.04),  # GUI's value divided by 100
    "num_ellipse_pts" / Default(Int, 72),
    "arrow" / Default(
        Struct(
            "style" / Int,
            "attachment" / Int,
            "size" / Double,
            "angle" / Double,  # GUI's value converted to radian
            "macro_function" / TecStr,

        ), dict(
            stype=0,
            attachment=0,
            size=0.05,
            angle=0.0,
            macro_function="",
            data_type=1,
            clipping=0
        )
    ),
    "data_type" / Default(Int, 1),  # 1=Float, 2=Double
    "clipping" / Default(Int, 0),
    "__integrity__" / Check(this.type == 0),
    "geom" / IfThenElse(
        this.type == 0,
        Struct(
            "num_polylines" / Rebuild(Int, len_(this.lines)),
            "lines" / Array(
                this.num_polylines,
                Struct(
                    "num_points" / Rebuild(Int, len_(this.x)),
                    "x" / Array(this.num_points, IfThenElse(this._._.data_type == 1, Float, Double)),
                    "y" / Array(this.num_points, IfThenElse(this._._.data_type == 1, Float, Double)),
                    StopIf(this._._.position_coord_sys != 4),
                    "z" / Array(this.num_points, IfThenElse(this._._.data_type == 1, Float, Double)),
                )
            )
        ),
        Error
    )
)


def gen_zone_struct(nvar: int):
    return Struct(
        "offset_start" / Tell,
        "izone" / Index,
        Const(Float.build(299.0)),  # Zone marker
        "title" / Default(TecStr, ""),  # Zone name
        Const(Int.build(-1)),  # Parent Zone
        "time_strand" / Default(Int, -2),  # StrandID
        "solution_time" / Default(Double, 0.0),  # Solution Time
        Const(Int.build(-1)),  # Default Zone Color
        "zone_type" / Default(Int, ZoneType.ORDERED),  # Zone Type
        "has_var_loc" / Default(Int, 1),  # Var Location = 1
        "var_loc" / Default(If(this.has_var_loc == 1, Array(nvar, Int)), [VarLoc.Node] * nvar),  # Var Loc * nvar_file
        Const(Int.build(0)),  # raw local 1-to-1
        Const(Int.build(0)),  # mcs user-defined face = 0,
        "ijk" / Default(If(this.zone_type == ZoneType.ORDERED, Array(3, Int)), None),
        "num_pts" / Default(If(this.zone_type != ZoneType.ORDERED, Int), None),
        "num_faces" / Default(If(this.zone_type == ZoneType.FEPOLYGON or this.zone_type == ZoneType.FEPOLYHEDRON, Int), None),
        "face_nodes" / Default(If(this.zone_type == ZoneType.FEPOLYGON or this.zone_type == ZoneType.FEPOLYHEDRON, Int), None),
        "boundary_faces" / Default(If(this.zone_type == ZoneType.FEPOLYGON or this.zone_type == ZoneType.FEPOLYHEDRON, Int), None),
        "boundary_connections" / Default(If(this.zone_type == ZoneType.FEPOLYGON or this.zone_type == ZoneType.FEPOLYHEDRON, Int), None),
        "num_elems" / Default(If(this.zone_type != ZoneType.ORDERED, Int), None),
        "cell_dim" / Default(If(this.zone_type != ZoneType.ORDERED, Array(3, Int)), None),
        "__integrity__" / Check(lambda this: bool(this.ijk is None) != bool(this.num_elems is None)),
        "aux_vars" / Default(GreedyRange(
            Struct(
                StopIf(Peek(Int) == 0),
                Const(Int.build(1)),
                "name" / TecStr,
                "format" / Default(Int, 0),
                "value" / TecStr,
            ),
        ), []),
        Const(Int.build(0)),
        "offset_end" / Tell,
    )


def gen_data_struct(variables, zone, read_data_=True):
    nvar = len(variables)
    return Struct(
        "offset_start" / Tell,
        "izone" / Index,
        Const(Float.build(299.0)),
        "data_type" / Default(Array(nvar, Int), [VarType.Float] * nvar),
        "has_passive_var" / Default(Int, 1),
        "passive_var" / Default(If(this.has_passive_var == 1, Array(nvar, Int)), [0] * nvar),
        "has_shared_var" / Default(Int, 1),
        "shared_var" / Default(If(this.has_shared_var == 1, Array(nvar, Int)), [-1] * nvar),
        "shared_connectivity" / Default(Int, -1),
        "ivar_zone" / Computed(lambda this: find_var_zone(this.has_passive_var, this.passive_var, this.has_shared_var, this.shared_var, nvar)),
        "nvar_zone" / Computed(lambda this: len(this.ivar_zone)),
        "min_max" / Array(this.nvar_zone, Array(2, Double)),
        "data" / Array(
            this.nvar_zone,
            Struct(
                "offset_start" / Tell,
                "ivar" / Computed(lambda this: this._.ivar_zone[this._index]),
                "variable" / Computed(lambda this: variables[this.ivar]),
                "data_type" / Computed(lambda this: this._.data_type[this.ivar]),
                "izone" / Computed(lambda this: this._.izone),
                "varloc" / Computed(lambda this: zone.var_loc[this.ivar] if 'has_var_loc' in zone and zone.has_var_loc == 1 else VarLoc.Node),
                "min" / Computed(lambda this: this._.min_max[this._index][0]),
                "max" / Computed(lambda this: this._.min_max[this._index][1]),
                "value" / TecplotMatrix(
                    lambda this: VarTypeConstruct[this.data_type],
                    cell_centered=lambda this: False if 'zone_type' not in zone or zone.zone_type != ZoneType.ORDERED else this.varloc == VarLoc.CellCentered,
                    ijk=lambda this: zone.ijk if zone.zone_type == ZoneType.ORDERED else [
                        zone.num_pts if this.varloc == VarLoc.Node else zone.num_elems,
                        1, 1],
                    discard=bool(not read_data_)
                ),
                "offset_end" / Tell
            )
        ),
        "offset_end" / Tell
    )


Logger = logging.getLogger(__name__)
Logger.setLevel(logging.INFO)
Handler = logging.StreamHandler(sys.stdout)
Handler.setLevel(logging.INFO)
Logger.addHandler(Handler)

class TecplotFile(construct.Container):
    """
    Tecplot Handler
    """

    def __init__(self, filePath: str, read_data=True):
        import time
        super().__init__()
        self.file = filePath
        self.read_data = read_data
        self.compressed = False
        self.open = open
        self.has_data = False

        with open(self.file, 'rb') as f:
            b = f.read(8)
            if b[:2] == b'\x1f\x8b':
                self.compressed = True
                self.open = gzip.open
            elif b != b'#!TDV112':
                raise ValueError(f'file {self.file} is not a valid Tecplot binary file')

        float_peek = Peek(Float)
        start_time = time.time()
        with self.open(self.file, 'rb') as f:
            self.update(TecHeader.parse_stream(f))
            reachingEOHM = False
            self.has_dataset_aux = False
            zone_struct = gen_zone_struct(self.nvar)
            while not reachingEOHM:
                marker = float_peek.parse_stream(f)
                if marker == 399.0:
                    Peek(TecGeom).parse_stream(f)
                    self.geometries = GreedyRange(TecGeom).parse_stream(f)
                elif marker == 299.0:
                    Peek(zone_struct).parse_stream(f)
                    self.zones = GreedyRange(zone_struct).parse_stream(f)
                elif marker == 799.0:
                    Peek(TecDatasetAux).parse_stream(f)
                    self._dataset_aux = GreedyRange(TecDatasetAux).parse_stream(f)
                    self.has_dataset_aux = True
                elif marker == 357.0:
                    Const(Float.build(357.0)).parse_stream(f)
                    self.has_data = True
                    break

            self.nzones = len(self.zones)
            self.data = ListContainer()
            for iz, z in enumerate(self.zones):
                self.data.append(gen_data_struct(self.variables, z, self.read_data).parse_stream(f))
            if len(self.data) == 0:
                raise ValueError("Fail to parse data")
        Logger.warn(f"Finish loading {filePath:s} with {self.nzones} zones in {time.time() - start_time:f}(s)")
        if self.has_dataset_aux:
            self.dataset_aux = construct.Container()
            for i in self._dataset_aux:
                if i.name in self.dataset_aux:
                    raise ValueError(f'Duplicated dataset aux {i}')
                if ',' in i.value:
                    self.dataset_aux[i.name] = i.value.split(',')
                else:
                    self.dataset_aux[i.name] = i.value

    def get_data(self, izone: int, ivar: int):
        import time
        """
            Get the data from izone for variable ivar
            izone and ivar should start from 0
        """
        izone = self.nzones + izone if izone < 0 else izone
        ivar = self.nvar + ivar if ivar < 0 else ivar
        offset = None
        d = self.data[izone]
        if ivar in d.ivar_zone:
            if self.read_data or d.data[d.ivar_zone.index(ivar)].value is not None:
                return d.data[d.ivar_zone.index(ivar)].value.copy()
            else:
                offset = d.data[d.ivar_zone.index(ivar)].offset_start

        if d.has_shared_var and d.shared_var[ivar] != -1:
            jzone = d.shared_var[ivar]
            return self.get_data(jzone, ivar)

        # It's a passive variable
        z = self.zones[izone]
        zt = z.zone_type
        vt = VarLoc.Node if not z.has_var_loc else z.var_loc[ivar]
        vdt = np.dtype(VarTypeConstruct[d.data_type[ivar]].fmtstr)
        if zt == ZoneType.ORDERED:
            shape = squeeze_ijk(z.ijk)
            if vt == VarLoc.CellCentered:
                shape = [i - 1 for i in shape if i > 1]
        else:
            shape = [z.num_elems if vt == VarLoc.CellCentered else z.num_pts]

        if offset is None:
            return np.zeros(shape, dtype=vdt)
        else:
            s = TecplotMatrix(VarTypeConstruct[d.data_type[ivar]],
                              cell_centered=False if z.zone_type != ZoneType.ORDERED else vt == VarLoc.CellCentered,
                              ijk=z.ijk if z.zone_type == ZoneType.ORDERED else [
                z.num_pts if vt == VarLoc.Node else z.num_elems,
                1, 1],
            )
            start_time = time.time()
            vname = d.data[d.ivar_zone.index(ivar)].variable
            with self.open(self.file, 'rb') as f:
                f.seek(offset)
                d.data[d.ivar_zone.index(ivar)].value = s.parse_stream(f)
            Logger.warning(f"Finish on-demand loading of Zone {izone:d} Variable \"{vname}\" in {time.time() - start_time:f} (s)")
            return d.data[d.ivar_zone.index(ivar)].value.copy()

    def get_solution_time(self, izone: int):
        return self.zones[izone].solution_time

    def get_min(self, izone: int, ivar: int):
        return self.data[0].data[-1].min

    def get_max(self, izone: int, ivar: int):
        return self.data[0].data[-1].max

    def get_dataset_aux(self):
        if self.has_dataset_aux:
            return self.dataset_aux
        else:
            return construct.Container()


if __name__ == "__main__":
    input_file = 'qPot_trj.tec'
    tec = TecplotFile(input_file, read_data=False)  # read data on demand

    # print a summery of the file
    print(tec)

    # get variable names
    print(tec.variables)

    # get zone info
    print(tec.zones[0])

    # get data for first zone and first variable
    data = tec.get_data(0, 0)  # via API

    # get max value of the last variable of the first zone
    max_val = tec.get_max(0, -1)
