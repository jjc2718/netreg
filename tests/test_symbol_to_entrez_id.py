import math
import unittest
import numpy as np

import sys; sys.path.append('.')
import config as cfg
from utilities.symbol_to_entrez_id import symbol_to_entrez_id

class SymbolConversionTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SymbolConversionTest, self).__init__(*args, **kwargs)

    def test_symbol_alias(self):
        """Test mapping symbols by their aliases.

        This tests the case where a symbol doesn't map directly to an
        Entrez ID, but it has an alias which does.
        """
        # CCDC83 has an exact match in MyGene
        # DDX26B has an alias (INTS6L) which has a match in MyGene
        symbols = ['DDX26B', 'CCDC83']
        gene_map = symbol_to_entrez_id(symbols)
        assert len(list(gene_map.keys())) == len(symbols)
        assert 'N/A' not in list(gene_map.values())

    def test_nan_entrez_id(self):
        """Test the case where a symbol is found in the DB, but without an ID.

        This can happen for either the symbol proper or one of its aliases,
        so we need to test both cases.
        """
        # BRAF is a well-studied gene which has an exact match
        # LILRP2 is a pseudogene which MyGene doesn't map to any Entrez ID
        # with high confidence
        # ILT10 is an alias for LILRP2, so should have the same effect
        # (no high-confidence mapping)
        symbols = ['BRAF', 'ILT10', 'LILRP2']
        gene_map = symbol_to_entrez_id(symbols)
        assert len(list(gene_map.keys())) == len(symbols)
        assert 'N/A' in list(gene_map.values())
        # previous versions of the code would read this case as NaN (which
        # causes issues downstream), so we should make sure there are no NaNs.
        # These should have value (string) 'N/A' instead.
        non_na_to_float = [math.isnan(float(l)) for l in gene_map.values()
                                                if l != 'N/A']
        assert sum(non_na_to_float) == 0

