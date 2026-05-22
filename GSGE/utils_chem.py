import copy
import logging
from multiprocessing import shared_memory
from typing import Union, List, Tuple, Any, Optional, Dict, Generator
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem, rdFMCS
from rdkit.Chem import MolToSmiles, MolFromSmiles, MolFromSmarts, SanitizeMol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol as RdkitMol

# scipy hierarchy clustering
from scipy.cluster.hierarchy import cophenet, cut_tree
from scipy.spatial.distance import pdist

# SKlearn metrics
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations, islice
import math
import os
import pickle

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.info")
# print(f"rdkit {rdkit.__version__}")

N_WORKERS = 20
BATCH_SIZE = 10000
string_types = (type(b""), type(""))


# Define a custom function to aggregate the columns
def custom_agg(x):
    if len(x) > 1:
        unique = x.nunique()
        if unique == 1:
            return x.iloc[0]
        else:
            return x.dropna().unique().tolist()
    else:
        return x.iloc[0]


def check_na(
    df, cols: Union[List[str], str] = "smiles", nan_dup_source="", logger=None
):
    """
    Check for NaN values in the specified column(s) of the given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check for NaN values.
    cols : Union[List[str], str], default='smiles'
        A list of column names or a single column name to check for NaN values.
    nan_dup_source : str, optional
        Source of NaN values in the DataFrame.
    logger : Logger, optional
        Logger object to use for logging.

    Returns:
    --------
    df : pandas.DataFrame
        A copy of the DataFrame with rows containing NaN values in the specified columns removed.
    df_nan : pandas.DataFrame
        A copy of the DataFrame with rows containing NaN values in the specified columns.

    Raises:
    -------
    AssertionError :
        If any column from `cols` is not present in the DataFrame.

    Example:
    --------
    # >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 4, 5], 'C': [7, 8, 9]})
    # >>> df_clean, df_nan = check_na(df, cols=['A', 'B'], nan_dup_source='original')
    # >>> print(df_clean)
       A    B  C
    0  1.0  4.0  7
    1  2.0  5.0  8
    # >>> print(df_nan)
         A   B  C nan_dup_source
    0  NaN NaN  9       original
    """
    if not isinstance(cols, list):
        cols = [cols]

    assert all([col in df.columns for col in cols]), (
        f"One or more columns from `cols` for checking NaNs, not in the dataframe"
        f"The dataframe cols are {df.columns}"
    )

    try:
        na_mask = df[cols].isna().any(axis=1)
        df_nan = df[na_mask].copy()
        df = df[~na_mask].copy()
        if nan_dup_source and (na_mask.sum() > 0):
            # (nan_df.shape[0] > 0):
            # (na_mask.sum() > 0):
            df_nan.loc[:, "nan_dup_source"] = nan_dup_source

        if logger:
            logger.info(
                f"Checked the {cols} column(s) for NaN values "
                f"Number of NaN values: {df_nan.shape[0]}"
            )
    except Exception as e:
        df_nan = pd.DataFrame(columns=df.columns)  # empty dataframe
        logger.error(
            f"Unexpected Error while checking for NaN empty values in {cols} in check_na() function: {e}"
        )

    return df, df_nan


def check_duplicates(
    df: pd.DataFrame,
    cols: Union[List[str], str],
    drop: bool = True,
    sorting_col: str = "",
    keep: Union[bool, str] = "first",
    nan_dup_source: str = "",
    logger=None,
):
    """
    Check for duplicates in the specified column(s) of the given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check for duplicates.
    cols : Union[List[str], str]
        A list of column names or a single column name to check for duplicates.
    drop : bool, optional, default=True
        If True, drop the duplicate rows.
    sorting_col : str, optional, default=""
        The column by which to sort the duplicates DataFrame.
    keep : Union[bool, str], optional, default='first'
        If drop is True, this parameter indicates which duplicate values to keep.
        If keep is 'first', the first occurrence of the duplicate value is kept.
        If keep is 'last', the last occurrence of the duplicate value is kept.
        If keep is False, all occurrences of the duplicate value are dropped.
    nan_dup_source : str, optional, default=""
        Source of duplicate values in the DataFrame.
    logger : Logger, optional
        Logger object to use for logging.

    Returns:
    --------
    df : pandas.DataFrame
        A copy of the DataFrame with duplicate rows removed if drop is True.
    df_dup : pandas.DataFrame
        A copy of the DataFrame with only duplicate rows if any.

    Raises:
    -------
    AssertionError :
        If any column from `cols` is not present in the DataFrame.
        If sorting_col is not present in the DataFrame.
        If keep is not one of 'first', 'last', or False.

    Example:
    --------
    # >>> df = pd.DataFrame({'A': [1, 2, 2], 'B': [4, 5, 4], 'C': [7, 8, 9]})
    # >>> df_clean, df_dup = check_duplicates(df, cols=['A', 'B'], keep='last')
    # >>> print(df_clean)
       A  B  C
    0  1  4  7
    2  2  4  9
    # >>> print(df_dup)
       A  B  C
    1  2  5  8
    """
    # df = df[[x, y]].
    if not isinstance(cols, list):
        cols = [cols]
    assert all([col in df.columns for col in cols]), (
        "One or more columns from `cols` for checking duplicates, not in the dataframe"
        f"The dataframe cols are {df.columns}"
    )
    assert sorting_col in df.columns or sorting_col == "", (
        "sorting_col not in the dataframe" f"The dataframe cols are {df.columns}"
    )
    assert keep in [
        "first",
        "last",
        False,
    ], "keep should be either 'first', 'last' or False"

    try:
        dup_mask = df.duplicated(subset=cols, keep=False)
        df_dup = df[dup_mask].copy()
        if nan_dup_source and dup_mask.sum() > 0:
            df_dup.loc[:, "nan_dup_source"] = nan_dup_source

        if sorting_col and (not df_dup.empty):
            s_cols = cols + [sorting_col]  # WRONG: s_cols = cols.append(sorting_col)
            df_dup = df_dup.sort_values(by=s_cols, ascending=True)

        if drop:
            # now we get the ones to keep
            to_keep = df_dup.drop_duplicates(subset=cols, keep=keep)
            # now we drop the ones to keep from the duplicates dataframe
            to_drop = df_dup.drop(to_keep.index)
            df = df.drop(to_drop.index)

        if logger:
            logger.info(
                f"Checked the {cols} column(s) for duplicates "
                f"Number of ALL duplicates: {df_dup.shape[0]}"
            )

    except Exception as e:
        df_dup = pd.DataFrame(columns=df.columns)  # empty dataframe
        logger.error(
            f"Unexpected Error while checking for duplicates in {cols} in check_duplicates() function: {e}"
        )

    return df, df_dup


def check_nan_duplicated(
    df,
    cols_nan: Union[List[str], str] = "smiles",
    cols_dup: Union[List[str], str] = "smiles",
    nan_dup_source="",
    drop=True,
    sorting_col="",
    keep="first",
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Check for NaN and duplicated values in a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to check for NaN and duplicated values.
    cols_nan : Union[List[str], str], optional
        The column(s) to check for NaN values, by default "smiles".
    cols_dup : Union[List[str], str], optional
        The column(s) to check for duplicated values, by default "smiles".
    nan_dup_source : str, optional
        The source of NaN or duplicated values, by default "".
    drop : bool, optional
        Whether to drop the duplicated values, by default True.
    sorting_col : str, optional
        The column used for sorting in the case of duplicated values, by default "".
    keep : Union[bool, str], optional
        Whether to keep the first or last occurrence of duplicated values, or False to drop all, by default "first".
    logger : logger object, optional
        Logger object for logging the results of the check, by default None.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the filtered DataFrame (without NaN or duplicated values), a DataFrame containing the rows
        with NaN values, and a DataFrame containing the duplicated rows.
    """
    # NaN Check
    df_filtered, df_nan = check_na(df, cols_nan, nan_dup_source, logger)
    # Duplicates Check
    df_filtered, df_dup = check_duplicates(
        df_filtered,
        cols_dup,
        drop=drop,
        sorting_col=sorting_col,
        keep=keep,
        nan_dup_source=nan_dup_source,
        logger=logger,
    )

    return df_filtered, df_nan, df_dup


def rdkit_standardize(smi, logger=None, suppress_exception=False):
    """
    Applies a standardization workflow to a SMILES string.

    Parameters:
    -----------
    smi : str
        The input SMILES string to standardize.

    logger : logging.Logger, optional
        A logger object to log error messages. Default is None.

    suppress_exception : bool, optional
        A boolean flag to suppress exceptions and return the original SMILES string if an error
        occurs during standardization. If False, an exception is raised or logged, depending on the value of logger.
        Default is True.

    Returns:
    --------
    str
        The standardized SMILES string.

    Raises
    ------
    TypeError
        If check_smiles_type is True and the input is not a string.
    StandardizationError
        If an unexpected error occurs during standardization and suppress_exception is False.
        The error message is logged or raised, depending on the value of logger.

    Notes:
    ------
    This function applies the following standardization steps to the input SMILES string:

    1. Functional Groups Normalization: The input SMILES string is converted to a molecule object,
    and any functional groups present are normalized to a standard representation.
    2. Sanitization: The molecule is sanitized, which involves performing various checks
    and corrections to ensure that it is well-formed.
    3. Neutralization: Any charges on the molecule are neutralized.
    4. Parent Tautomer: The canonical tautomer of the molecule is determined.

    This function uses the RDKit library for performing standardization.
    implementation source:
    https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    """
    if smi is None:
        return None
    og_smiles = copy.deepcopy(smi)
    try:
        # Functional Groups Normalization
        mol = MolFromSmiles(smi)
        mol.UpdatePropertyCache(strict=False)
        SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES
            ),
        )
        mol = rdMolStandardize.Normalize(mol)

        # Neutralization
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(rdMolStandardize.FragmentParent(mol))

    except Exception as e:
        if logger:
            logger.error(f"StandardizationError: {e} for {og_smiles}")
        if suppress_exception:
            return og_smiles
        else:
            return None

    return MolToSmiles(mol)


def remove_stereo_rdkit_molecule(
    mol: RdkitMol,
) -> Optional[RdkitMol]:
    try:
        Chem.RemoveStereochemistry(mol)
        return mol

    except Exception as e:
        raise ValueError(
            f"Removing Stereochemistry failed with the following error {e}"
        )


def neutralize_rdkit_molecule(
    mol: RdkitMol,
) -> Optional[RdkitMol]:
    try:
        pattern = MolFromSmarts(
            "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4]),-1!$([*]~[1+,2+,3+,4+])]"
        )
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]

        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                h_count = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(h_count - chg)
                atom.UpdatePropertyCache()

        return mol

    except Exception as e:
        raise ValueError(f"Neutralization failed with the following error {e}")


def remove_isotopes_rdkit_molecule(
    mol: RdkitMol,
) -> Optional[RdkitMol]:
    try:
        atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
        for atom, isotope in atom_data:
            # restore original isotope values
            if isotope:
                atom.SetIsotope(0)
        Chem.RemoveHs(mol)
        return mol

    except Exception as e:
        raise ValueError(f"Removing Isotope failed with the following error {e}")


def standardize(
    smiles: Optional[str],
    logger: Optional[logging.Logger] = None,
    suppress_exception: bool = True,
    remove_stereo: bool = False,
) -> Optional[str]:
    """
    Standardizes a given SMILES string using RDKit.

    Parameters
    ----------
    smiles : str, optional
        A SMILES string to be standardized. If None, returns None.
    remove_stereo : bool, optional
        A boolean flag to remove stereochemistry information if True. Default is False.
    logger : logging.Logger, optional
        A logger object to log error messages. Default is None.
    suppress_exception : bool, optional
        A boolean flag to suppress exceptions and return the original SMILES string if an error
        occurs during standardization. If False, an exception is raised or logged, depending on the value of logger.
        Default is True.

    Returns
    -------
    str or None
        The standardized SMILES string or None, depending on the value of suppress_exception and whether an exception
        occurs. If an exception occurs and suppress_exception is True, the original SMILES string is returned.
    """
    if smiles is None:
        return None
    og_smiles = copy.deepcopy(smiles)
    smiles_inter = None

    try:
        smiles = smiles.split("|")[0].split("{")[0].strip()
        mol = MolFromSmiles(smiles)
        if mol is None:
            return None

        smiles_inter = MolToSmiles(mol, canonical=True)

        if remove_stereo:
            mol = remove_stereo_rdkit_molecule(mol)
        mol = neutralize_rdkit_molecule(mol)
        mol = remove_isotopes_rdkit_molecule(mol)
        # For Sanity Double Check
        smiles = MolToSmiles(mol, canonical=True)
        mol = MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smiles

    except Exception as e:
        if logger:
            logger.error(f"StandardizationError: {e} for {og_smiles}")
        if suppress_exception:
            return smiles_inter
        else:
            return None

def standardize_wrapper(args):
    """
    Wrapper function for the standardize function to be used with the concurrent.futures.ProcessPoolExecutor.
    """
    return standardize(*args)


def rdkit_standardize_wrapper(args):
    """
    Wrapper function for the rdkit_standardize function to be used with the concurrent.futures.ProcessPoolExecutor.
    """
    return rdkit_standardize(*args)


def parallel_standardize(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    logger=None,
    suppress_exception=True,
    rd_standardize=False,
):
    # standardizing the SMILES in parallel
    standardizer = rdkit_standardize_wrapper if rd_standardize else standardize_wrapper
    unique_smiles = df[smiles_col].unique()
    args_list = [(smi, logger, suppress_exception) for smi in unique_smiles]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(
            tqdm(
                executor.map(standardizer, args_list),
                total=len(args_list),
                desc="Standardizing Unique SMILES",
            )
        )
    standardized_result = {smi: result for smi, result in zip(unique_smiles, results)}

    # Apply the standardized result to the dataframe
    df[smiles_col] = df[smiles_col].map(standardized_result)

    return df


def standardize_df(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    other_dup_col: Union[List[str], str] = None,
    sorting_col: str = "",
    drop: bool = True,
    keep: Union[bool, str] = "last",
    logger=None,
    suppress_exception=True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies a standardization workflow to the 'smiles' column of a pandas dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe, which should contain a 'smiles' column.

    smiles_col : str, optional
        The name of the column containing the SMILES strings to standardize. Default is 'smiles'.

    other_dup_col : str or list of str, optional
        The name of the column(s) containing other information that should be kept for duplicate SMILES.
        If None, no other columns are kept. Default is None.

    sorting_col : str, optional
        The name of the column to sort the dataframe by before standardization. If None, the dataframe is not sorted.
        Default is None.

    drop : bool, optional
        A boolean flag to drop the rows with NaN SMILES before standardization. Default is True.

    keep : bool or str, optional
        A boolean flag to keep the first or last duplicate SMILES. If True, the first duplicate is kept.
        If False, the last duplicate is kept. If 'aggregate', the duplicates are aggregated into a list.
        Default is 'last'.

    logger : logging.Logger, optional
        A logger object to log error messages. Default is None.

    suppress_exception : bool, optional
        A boolean flag to suppress exceptions and return the original SMILES string if an error
        occurs during standardization. If False, an exception is raised or logged, depending on the value of logger.
        Default is True.

    Returns:
    --------
    pandas.DataFrame
        A new dataframe with the 'smiles' column replaced by the standardized versions.

    Notes:
    ------
    This function applies the `standardize` function to each SMILES string in the 'smiles' column
    of the input dataframe,
    and replaces the column with the standardized versions.
    """
    if keep == "aggregate":
        keep = False
        aggregate = True
    else:
        aggregate = False

    if other_dup_col:
        if not isinstance(other_dup_col, list):
            other_dup_col = [other_dup_col]
        cols_dup = [smiles_col, *other_dup_col]
    else:
        cols_dup = smiles_col

    # checking NaN & duplicate before standardization
    df_filtered, df_nan_before, df_dup_before = check_nan_duplicated(
        df=df,
        cols_nan=smiles_col,
        cols_dup=cols_dup,  # [smiles_col, *other_dup_col],
        nan_dup_source="smiles_before_std",
        drop=drop,
        sorting_col=sorting_col,
        keep=keep,
        logger=logger,
    )
    if logger:
        logger.debug(
            f"BEFORE SMILES standardization, The number of filtered-out NaN values"
            f"is: {df_nan_before.shape[0]} NaN values"
            f"While The number of points that were found to be duplicates"
            f"is: {df_dup_before.shape[0]} duplicated rows"
        )

    df_filtered = parallel_standardize(
        df_filtered, smiles_col, logger, suppress_exception
    )

    df_filtered, df_nan_after, df_dup_after = check_nan_duplicated(
        df=df_filtered,
        cols_nan=smiles_col,
        cols_dup=cols_dup,  # [smiles_col, *other_dup_col],
        nan_dup_source="smiles_after_std",
        drop=drop,
        sorting_col=sorting_col,
        keep=keep,
        logger=logger,
    )

    if logger:
        logger.info(
            f"After SMILES standardization, The number of additional NaN values (failed standardization) "
            f"is: {df_nan_after.shape[0]} NaN values"
            f"While The number of points that were found to be duplicates after standardization "
            f"is: {df_dup_after.shape[0]} duplicated rows"
        )

    # concat the nan and dup dataframes
    df_nan = pd.concat([df_nan_before, df_nan_after])
    df_dup = pd.concat([df_dup_before, df_dup_after])

    if aggregate:
        # aggregate the duplicates
        df_dup = (
            df_dup.groupby(smiles_col, as_index=False).agg(custom_agg).reset_index()
        )

    return df_filtered, df_nan, df_dup


def save_npy_file(array_data: np.ndarray, filepath: str) -> None:
    """
    Saves a numpy array to a file with a .npy extension.

    Parameters:
    -----------
    array_data : np.ndarray
        The numpy array to save.
    filepath : str
        The path where the file will be saved.

    Returns:
    --------
    None
    """
    # Save the array to the specified filepath
    np.save(filepath, array_data)


def load_npy_file(filepath: Union[str, Path]) -> np.ndarray:
    """
    Loads a .npy file into a numpy array.

    Parameters:
    -----------
    filepath : str
        The path to the file to load.

    Returns:
    --------
    np.ndarray
        The loaded numpy array.
    """
    # Load the file as a numpy array
    return np.load(filepath)

def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    Saves data to a pickle file.

    Parameters:
    -----------
    data : Any
        The data to be saved.
    file_path : str
        The file path where the pickle file will be stored.

    Returns:
    --------
    None
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Loads data from a pickle file.

    Parameters:
    -----------
    filepath : str
        The path to the pickle file.

    Returns:
    --------
    Any
        The data loaded from the pickle file.
    """
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {e}")


def save_df(
    df: pd.DataFrame,
    file_path: Union[str, None] = None,
    output_path: str | Path = "./",
    filename: str = "exported_file",
    ext: str = "csv",
    **kwargs,
) -> None:
    """
    Exports a DataFrame to a specified file format.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be exported.
    file_path : str or None, optional
        Full file path for export (default: None).
    output_path : str, optional
        The directory path for exporting files (default: "./").
    filename : str, optional
        The filename to use if file_path is not specified (default: "exported_file").
    ext : str, optional
        The file extension to use for saving (default: "csv").
    kwargs : dict
        Additional arguments for file export.

    Returns:
    --------
    None
    """
    if df.empty:
        return logging.warning("DataFrame is empty. Nothing to export.")

    if file_path:
        path_obj = Path(file_path)
        output_path = path_obj.parent
        # filename = path_obj.stem
        ext = path_obj.suffix.lstrip(".")
    else:
        if not filename.endswith(ext):
            filename = f"{filename}.{ext}"
        # file_path = Path(output_path) / filename
        file_path = os.path.join(output_path, filename)

    # os.makedirs(output_path, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if ext == "csv":
        df.to_csv(file_path, index=False, **kwargs)
    elif ext == "parquet":
        df.to_parquet(file_path, index=False, **kwargs)
    elif ext == "feather":
        df.to_feather(file_path, **kwargs)
    elif ext == "pkl":
        df.to_pickle(file_path, **kwargs)
    elif ext == "xlsx":
        df.to_excel(file_path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    logging.info(f"DataFrame exported to {file_path}")


def load_df(input_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Loads a DataFrame from a specified file format.

    Parameters:
    -----------
    input_path : str or Path
        The file path to load.
    kwargs : dict
        Additional arguments for file loading.

    Returns:
    --------
    pd.DataFrame
        The loaded DataFrame.
    """
    # we want to get the extension from the Path object
    if isinstance(input_path, string_types):
        input_path = Path(input_path)
    ext = input_path.suffix.lstrip(".")
    if ext == "csv":
        return pd.read_csv(input_path, **kwargs)
    elif ext == "parquet":
        return pd.read_parquet(input_path, **kwargs)
    elif ext == "pkl":
        return pd.read_pickle(input_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension for loading: {ext} provided")



def tanimoto_mcs(smi1: str, smi2: str) -> float:
    """
    Computes the Tanimoto similarity based on the Maximum Common Substructure (MCS) between two molecules.

    Parameters:
    -----------
    smi1 : str
        SMILES string of the first molecule.
    smi2 : str
        SMILES string of the second molecule.

    Returns:
    --------
    float
        Tanimoto similarity score based on MCS.
    """
    # reading smiles of two molecules and create molecule
    # Replace '*1' with '*' to avoid RDKit ring closure conflicts with attachment points
    m1 = Chem.MolFromSmiles(smi1.replace('*1', '*'))
    m2 = Chem.MolFromSmiles(smi2.replace('*1', '*'))
    if m1 is None or m2 is None:
        return 0.0
    mols = [m1, m2]

    # number heavy atoms of both molecules
    a = m1.GetNumAtoms()
    b = m2.GetNumAtoms()
    # print(a,b)
    # find heavy atoms in MCS
    r = rdFMCS.FindMCS(
        mols,
        ringMatchesRingOnly=True,
        bondCompare=Chem.rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        timeout=1,
    )
    c = r.numAtoms
    # print(c)
    if c < 0:
        c = 0
    mcs_tani = c / (a + b - c)
    # get MCS smart
    # mcs_sm = r.smartsString
    return mcs_tani


def tanimoto_mcs_withH(smi1: str, smi2: str) -> float:
    """
    Computes the Tanimoto similarity based on the Maximum Common Substructure (MCS) between two hydrogen-included molecules.

    Parameters:
    -----------
    smi1 : str
        SMILES string of the first molecule.
    smi2 : str
        SMILES string of the second molecule.

    Returns:
    --------
    float
        Tanimoto similarity score based on MCS with hydrogen included.
    """
    # reading smiles of two molecules and create molecule
    # Replace '*1' with '*' to avoid RDKit ring closure conflicts with attachment points
    m1 = Chem.MolFromSmiles(smi1.replace('*1', '*'))
    m2 = Chem.MolFromSmiles(smi2.replace('*1', '*'))
    if m1 is None or m2 is None:
        return 0.0

    m1H = Chem.AddHs(m1)
    m2H = Chem.AddHs(m2)
    mols = [m1H, m2H]

    # number heavy atoms of both molecules
    a = m1H.GetNumAtoms()
    b = m2H.GetNumAtoms()
    # find heavy atoms in MCS
    r = rdFMCS.FindMCS(
        mols,
        ringMatchesRingOnly=True,
        bondCompare=Chem.rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        timeout=1,
    )
    c = r.numAtoms
    # print(c)
    if c < 0:
        c = 0
    mcs_tani = c / (a + b - c)
    # get MCS smart
    # mcs_sm = r.smartsString
    return mcs_tani


def tanimoto_mcs_wrapper(index_pair: Tuple[int, int], cid_list: List[str]) -> float:
    """
    Wrapper function for computing Tanimoto similarity using MCS.

    Parameters:
    -----------
    index_pair : Tuple[int, int]
        Index pair representing two molecules in a list.
    cid_list : List[str]
        List of SMILES strings corresponding to compound IDs.

    Returns:
    --------
    float
        Computed Tanimoto similarity score.
    """
    cid1, cid2 = index_pair
    return tanimoto_mcs(cid_list[cid1], cid_list[cid2])


def tanimoto_mcs_withH_wrapper(
    index_pair: Tuple[int, int], cid_list: List[str]
) -> float:
    """
    Wrapper function for computing Tanimoto similarity using MCS with hydrogen included.

    Parameters:
    -----------
    index_pair : Tuple[int, int]
        Index pair representing two molecules in a list.
    cid_list : List[str]
        List of SMILES strings corresponding to compound IDs.

    Returns:
    --------
    float
        Computed Tanimoto similarity score with hydrogen included.
    """
    cid1, cid2 = index_pair
    return tanimoto_mcs_withH(cid_list[cid1], cid_list[cid2])


def chunked_iterable(n: int, size: int) -> Generator[list[tuple[int, int]], Any, None]:
    """
    Generates chunks of index pairs for all unique molecule comparisons.

    Parameters:
    -----------
    n : int
        Total number of compounds.
    size : int
        Chunk size for processing pairs.

    Yields:
    --------
    List[Tuple[int, int]]
        List of index pairs in chunks.
    """
    iterable = combinations(range(n), 2)
    while True:
        chunk = list(islice(iterable, size))
        if not chunk:
            return
        yield chunk


def calculate_total_chunks(n_compounds: int, batch_size: int) -> int:
    """
    Calculates the total number of chunks required for pairwise similarity calculations.

    Parameters:
    -----------
    n_compounds : int
        Total number of compounds.
    batch_size : int
        Batch size for chunked processing.

    Returns:
    --------
    int
        Total number of chunks needed.
    """
    total_pairs = n_compounds * (n_compounds - 1) / 2
    total_chunks = math.ceil(total_pairs / batch_size)
    return total_chunks


def process_chunk(
    chunk: List[Tuple[int, int]],
    similarity_matrix: np.ndarray,
    cid_list: List[str],
    tanimoto_mcs_func: callable,
) -> None:
    """
    Processes a chunk of index pairs and updates the similarity matrix.

    Parameters:
    -----------
    chunk : List[Tuple[int, int]]
        List of index pairs for similarity computation.
    similarity_matrix : np.ndarray
        Preallocated similarity matrix.
    cid_list : List[str]
        List of SMILES strings.
    tanimoto_mcs_func : callable
        Function to compute Tanimoto similarity.

    Returns:
    --------
    None
    """
    # ThreadPoolExecutor avoids Windows spawn issues (BrokenProcessPool) that
    # occur with ProcessPoolExecutor in Jupyter notebooks. RDKit releases the
    # GIL for MCS computations, so threading still provides parallelism.
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(tanimoto_mcs_func, pair, cid_list): pair for pair in chunk
        }
        for future in as_completed(futures):
            similarity = future.result()
            i, j = futures[future]
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity


def save_similarity_matrix(
    matrix: np.ndarray, filename: str = "similarity_matrix.npy"
) -> None:
    """
    Saves the similarity matrix to a file.

    Parameters:
    -----------
    matrix : np.ndarray
        The similarity matrix to be saved.
    filename : str, optional
        Name of the file to save the matrix (default: "similarity_matrix.npy").

    Returns:
    --------
    None
    """
    np.save(filename, matrix)
    print(f"Similarity matrix saved to {filename}")


def hierarchical_clustering(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    batch_size: int = 10000,
    withH: bool = False,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Performs hierarchical clustering based on Maximum Common Substructure (MCS) similarity.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing SMILES strings.
    smiles_col : str, optional
        Column containing SMILES strings (default: "SMILES").
    batch_size : int, optional
        Size of each processing batch (default: 10000).
    withH : bool, optional
        Whether to include hydrogen in the similarity calculations (default: False).
    save_path : str, optional
        Path to save the similarity matrix (default: None).

    Returns:
    --------
    np.ndarray
        The computed similarity matrix.
    """
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_path) / "mcs.pkl.npy"

        # now checking if file exists
        if filepath.exists():
            print(f"Similarity matrix already exists at {filepath}")
            return load_npy_file(filepath)

    print(f"Chunk Size: {batch_size}")
    print(f"Number of Workers: {N_WORKERS}")

    cid_list = df[smiles_col].tolist()
    n_compounds = len(cid_list)
    print(f"Number of unique {smiles_col} for clustering: {df.shape[0]}")

    # Initialize the similarity matrix
    similarity_matrix = np.zeros((n_compounds, n_compounds), dtype="float16")
    # Calculate total chunks for tqdm progress bar
    total_chunks = calculate_total_chunks(n_compounds, batch_size)
    tanimoto_mcs_func = tanimoto_mcs_withH_wrapper if withH else tanimoto_mcs_wrapper
    # Process the pairs in chunks
    for chunk in tqdm(
        chunked_iterable(n_compounds, batch_size),
        desc="Calculating similarities in chunks",
        unit="chunk",
        total=total_chunks,
    ):
        process_chunk(chunk, similarity_matrix, cid_list, tanimoto_mcs_func)

    np.fill_diagonal(similarity_matrix, 1.0)

    # Save the similarity matrix to the specified path
    if save_path is not None:
        save_npy_file(similarity_matrix, filepath)
        # file_path = Path(save_path) / "mcs.npy"
        # save_similarity_matrix(similarity_matrix, save_path)

    return similarity_matrix


def form_linkage(
    X: np.ndarray,
    save_path: Optional[str] = None,
    calculate_cophenetic_coeff: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes hierarchical clustering linkage matrix using Ward's method.

    Parameters:
    -----------
    X : np.ndarray
        Pairwise similarity matrix.
    save_path : str, optional
        Path to save the linkage matrix (default: None).
    calculate_cophenetic_coeff : bool, optional
        Whether to calculate the cophenetic coefficient (default: True).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Processed similarity matrix and linkage matrix.
    """
    # start = time.time()
    n_rows, n_cols = X.shape
    upper_indices = np.triu_indices(n_rows, 1)
    x_dist = 1 - X[upper_indices]
    X_ = 1 - X
    filepath = None
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_path) / "mcs_linkage.pkl.npy"
        # now checking if file exists
        if filepath.exists():
            print(f"Linkage matrix already exists at {filepath}")
            return X_, load_npy_file(filepath)

    # check if X_ and X2 are the same
    from fastcluster import linkage
    Z = linkage(x_dist, method="ward")

    # Z = linkage(X, method="ward") # TODO : save and load if existing
    if save_path is not None:
        save_npy_file(Z, filepath)
    if calculate_cophenetic_coeff:
        calculate_cophenet(X_, Z, save_path=save_path)
    return X_, Z


def calculate_cophenet(
    X: np.ndarray, Z: np.ndarray, save_path: Optional[str] = None
) -> float:
    """
    Computes the cophenetic correlation coefficient for a given linkage matrix.

    Parameters:
    -----------
    X : np.ndarray
        Pairwise similarity matrix.
    Z : np.ndarray
        Linkage matrix from hierarchical clustering.
    save_path : str, optional
        Path to save cophenetic coefficient data (default: None).

    Returns:
    --------
    float
        Cophenetic correlation coefficient.
    """

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        Pdist_path = Path(save_path) / "mcs_pdist.pkl.npy"
        Coph_dists_path = Path(save_path) / "mcs_coph_dists.pkl.npy"
        C_path = Path(save_path) / "mcs_c.pkl"
        # now checking if file exists
        if C_path.exists():
            print(f"Cophenetic Coefficient already exists at {C_path}")
            return load_pickle(C_path)
    else:
        Pdist_path, Coph_dists_path, C_path = None, None, None

    Pdist = pdist(X)
    c, coph_dists = cophenet(Z, Pdist)
    print("Cophenetic coefficient calculated: %0.4f" % c)
    if save_path is not None:
        save_npy_file(Pdist, Pdist_path)
        save_npy_file(coph_dists, Coph_dists_path)
        save_pickle(c, C_path)
    return c


def calculate_silhouette(
    k: int, shm_name: str, shape: Tuple[int, int], Z: np.ndarray
) -> Tuple[int, float]:
    """
    Computes the silhouette score for a given number of clusters.

    Parameters:
    -----------
    k : int
        Number of clusters.
    shm_name : str
        Shared memory name for accessing precomputed similarity matrix.
    shape : Tuple[int, int]
        Shape of the similarity matrix.
    Z : np.ndarray
        Linkage matrix for hierarchical clustering.

    Returns:
    --------
    Tuple[int, float]
        Number of clusters and corresponding silhouette score.
    """
    # Access the shared memory block
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    X_shared = np.ndarray(shape, dtype=np.float32, buffer=existing_shm.buf)

    cluster_labels = cut_tree(Z, n_clusters=k).flatten()
    silhouette_avg = silhouette_score(X_shared, cluster_labels, metric="precomputed")

    existing_shm.close()
    return k, silhouette_avg


def calculate_silhouette_helper(
    args: Tuple[int, str, Tuple[int, int], np.ndarray]
) -> Tuple[int, float]:
    """
    Wrapper function for parallel silhouette score computation.

    Parameters:
    -----------
    args : Tuple[int, str, Tuple[int, int], np.ndarray]
        Arguments including cluster count, shared memory name, similarity matrix shape, and linkage matrix.

    Returns:
    --------
    Tuple[int, float]
        Number of clusters and corresponding silhouette score.
    """
    k, shm_name, shape, Z = args
    return calculate_silhouette(k, shm_name, shape, Z)


def sil_K(
    X: np.ndarray, Z: np.ndarray, max_k: int = 500
) -> Tuple[List[int], List[float], int]:
    """
    Determines the optimal number of clusters based on silhouette scores.

    Parameters:
    -----------
    X : np.ndarray
        Pairwise similarity matrix.
    Z : np.ndarray
        Linkage matrix from hierarchical clustering.
    max_k : int, optional
        Maximum number of clusters to evaluate (default: 500).

    Returns:
    --------
    Tuple[List[int], List[float], int]
        List of cluster sizes, corresponding silhouette scores, and optimal cluster count.
    """
    # Create shared memory block for X
    X = np.array(X, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
    X_shared = np.ndarray(X.shape, dtype=np.float32, buffer=shm.buf)  # X.dtype
    np.copyto(X_shared, X)
    # Prepare arguments for the helper function
    args_list = [(k, shm.name, X.shape, Z) for k in range(2, max_k)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        # results = pool.starmap(calculate_silhouette, [(k, shm.name, X.shape, Z) for k in range(2, max_k)])
        # Initialize the tqdm progress bar
        results = []
        for result in tqdm(
            pool.imap_unordered(calculate_silhouette_helper, args_list),
            total=max_k - 2,
            desc="Calculating silhouette scores",
        ):
            results.append(result)

    # Clean up shared memory
    shm.close()
    shm.unlink()
    results.sort(key=lambda x: x[0])  # Ensure the results are sorted by k
    n_clu, sil = zip(*results)
    optimal_clu = n_clu[sil.index(max(sil))]
    print("Optimal number of clusters: ", optimal_clu)

    return list(n_clu), list(sil), optimal_clu


def plot_silhouette_analysis(
    cluster_counts: List[int],
    silhouette_scores: List[float],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plots the silhouette analysis for determining the optimal number of clusters.

    Parameters:
    -----------
    cluster_counts : List[int]
        List of cluster sizes.
    silhouette_scores : List[float]
        List of silhouette scores corresponding to each cluster size.
    output_path : str, optional
        Path to save the plot (default: None).

    Returns:
    --------
    None
    """
    # Initialize the plot
    fig = plt.figure(figsize=(12, 5), dpi=600)
    plt.rc("font", family="serif")
    plt.plot(cluster_counts, silhouette_scores)  # , label="MCS"
    # plt.scatter(cluster_counts, silhouette_scores, label="MCS")
    # # Plot each series of cluster counts vs. silhouette scores
    # for i in range(len(cluster_counts)):
    #     plt.scatter(cluster_counts[i], silhouette_scores[i], label=labels[i])

    # Adding plot details
    # plt.legend(loc="lower right", shadow=True, fontsize=16)
    plt.xlabel("Number of Clusters", fontsize=16)
    plt.ylabel("Average Silhouette Score", fontsize=16)

    # Show and save the plot
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            Path(output_path)
            / "Silhouette_analysis_for_determining_optimal_clusters_K.png",
            bbox_inches="tight",
        )


def plot_cluster_heatmap(
    data_matrix: np.ndarray, output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Generates a heatmap for hierarchical clustering results.

    Parameters:
    -----------
    data_matrix : np.ndarray
        Clustering similarity matrix.
    output_path : str, optional
        Path to save the heatmap image (default: None).

    Returns:
    --------
    None
    """
    if hasattr(data_matrix, "index"):
        yticklabels = data_matrix.index
    else:
        # For numpy arrays, create numeric labels for each row
        yticklabels = range(data_matrix.shape[0])
    # yticklabels = data_matrix.index
    plt.figure(figsize=(12, 30), dpi=600)
    plt.rc("font", family="serif", size=8)
    sns.set_style("white")

    # Generate the clustermap
    fig = sns.clustermap(
        data_matrix,
        method="ward",
        cmap="coolwarm",
        fmt="d",
        linewidth=0.5,
        xticklabels=False,
        yticklabels=yticklabels,
        figsize=(12, 20),
    )

    # Save the plot to the specified output path
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            Path(output_path) / "Heatmap_of_the_clustering.png",
            dpi=600,
            bbox_inches="tight",
        )

    # Show the plot
    # plt.show()


def clustering(
    df: pd.DataFrame,
    smiles_col: str = "scaffold",
    max_k: int = 500,
    optimal_k: Optional[int] = None,
    withH: bool = False,
    export_mcs_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Performs hierarchical clustering based on molecular similarity.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing molecular data.
    smiles_col : str, optional
        Column containing molecular representations (default: "scaffold").
    max_k : int, optional
        Maximum number of clusters to evaluate (default: 500).
    optimal_k : int, optional
        Predefined optimal number of clusters (default: None).
    withH : bool, optional
        Whether to consider hydrogen atoms in similarity calculations (default: False).
    export_mcs_path : str, optional
        Path to save processed clustering data (default: None).

    Returns:
    --------
    pd.DataFrame
        Clustered DataFrame with assigned cluster labels.
    """
    clustered_df_path = None
    if export_mcs_path:
        clustered_df_path = Path(export_mcs_path) / "clustered_df.pkl"
        if clustered_df_path.exists():
            print(f"Clustered DataFrame already exists at {clustered_df_path}")
            return load_pickle(clustered_df_path)
    # pre cleaning
    df_clean = df.copy()[[smiles_col]]
    # dropp duplicates to avoid self comparison and reset index
    df_clean.drop_duplicates(subset=smiles_col, keep="first", inplace=True)

    df_clean.dropna(subset=[smiles_col], inplace=True)
    # print(f"after nan drop: {df_clean.shape}")
    df_clean.reset_index(inplace=True, drop=True)

    # TODO : checking mcs file existing then loading it instead of recalculation
    mcs_np = hierarchical_clustering(
        df_clean,
        smiles_col=smiles_col,
        batch_size=BATCH_SIZE,
        withH=withH,
        save_path=export_mcs_path,
    )
    # Just for debugging
    # df_clean = df_clean[:1000]
    # mcs_np = mcs_np[:1000, :1000]
    # if export_mcs_path:
    #     Path(export_mcs_path).mkdir(parents=True, exist_ok=True)
    #     df_mcs.to_csv(Path(export_mcs_path) / "mcs_matrix.csv", index=True)
    # df_pair.to_csv(Path(export_mcs_path) / "scaffold_sim_pair.csv", index=True)
    mcs_x, mcs_z = form_linkage(
        mcs_np, save_path=export_mcs_path, calculate_cophenetic_coeff=True
    )
    max_k = min(max_k, df_clean[smiles_col].nunique())
    # max_k = df_clean[smiles_col].nunique ()
    print(f"Max number of clusters: {max_k}")
    if optimal_k is None:
        mcs_k, mcs_sil, optimal_k = sil_K(mcs_x, mcs_z, max_k=max_k)  # , max_k=max_k
        if export_mcs_path:
            fig_output_path = Path(export_mcs_path) / "mcs_figures"
            Path(fig_output_path).mkdir(parents=True, exist_ok=True)
            plot_silhouette_analysis(mcs_k, mcs_sil, output_path=fig_output_path)

            optimal_k_path = Path(export_mcs_path) / f"mcs_optimal_k.pkl"
            save_pickle(optimal_k, optimal_k_path)

            # saving the silhouette scores
            sil_scores_path = Path(export_mcs_path) / f"mcs_sil_scores.pkl"
            save_pickle(zip(mcs_k, mcs_sil), sil_scores_path)

        # plot_cluster_heatmap(mcs_np, output_path=fig_output_path) # TAKES SO MUCH TIME
    # optimal_clu = 11974
    # df_clean["cluster"] = fcluster(mcs_z, optimal_clu, criterion="maxclust")
    print(f"Optimal number of clusters: {optimal_k}")
    df_clean["cluster"] = cut_tree(mcs_z, n_clusters=optimal_k).flatten()

    # now we map the cluster to the original dataframe
    df = pd.merge(df, df_clean, on=smiles_col, how="left", validate="many_to_many")
    df["cluster"] = df["cluster"].astype("Int64")
    if export_mcs_path:
        save_pickle(df, clustered_df_path)
    return df



############ EXAMPLE USAGE ############
# df_filtered, df_nan, df_dup = standardize_df(
#             df=df_filtered,
#             smiles_col="SMILES",
#             other_dup_col="accession",
#             drop=True,
#             sorting_col="Year",
#             keep="last",  # keeps latest year datapoints if duplicated
#             logger=logger,
#             suppress_exception=False,
#         )
