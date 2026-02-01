#!/usr/bin/env python3
import unittest
from unittest.mock import Mock, ANY, patch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.iuncld import iunCloud

class TestIuncloud(unittest.TestCase):
    def setUp(self):
        iunCloud.dserver = 'http://localhost:5188/'

    def test_check_bkignored(self):
        self.assertTrue(iunCloud.is_bk_ignored('BK0511'))

    def test_is_stock_blacked(self):
        self.assertFalse(iunCloud.is_stock_blacked('000001'))

    def test_to_be_divided(self):
        self.assertFalse(iunCloud.to_be_divided('000001'))

    def test_get_stock_bks(self):
        self.assertIsInstance(iunCloud.get_stock_bks('000001'), list)

    def test_get_bk_stocks(self):
        self.assertIsInstance(iunCloud.get_bk_stocks('BK0511'), list)

    def test_recent_zt(self):
        self.assertFalse(iunCloud.recent_zt('000001'))

    def test_topbks5(self):
        self.assertIsInstance(iunCloud.topbks5(), list)

    def test_get_hotstocks(self):
        self.assertIsInstance(iunCloud.get_hotstocks(), list)

    def test_get_stock_fflow(self):
        self.assertIsInstance(iunCloud.get_stock_fflow('000001'), list)

    def test_get_dailyzdt(self):
        self.assertIsInstance(iunCloud.get_dailyzdt(), list)

    def test_get_dailyztsteps_gt3(self):
        self.assertIsInstance(iunCloud.get_dailyztsteps_gt3(), list)

if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestIuncloud('test_get_dailyztsteps_gt3'))
    # unittest.TextTestRunner().run(suite)
