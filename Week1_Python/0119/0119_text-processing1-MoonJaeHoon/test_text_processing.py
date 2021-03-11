# -*- coding: utf8 -*-

import unittest
import random
import text_processing as tp


class TestTextProcessing(unittest.TestCase):
    def test_normalize(self):
        test_str = "This is an example."
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "this is an example.")

        test_str = "   EXTRA   SPACE   "
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "extra space")

        test_str = "THIS IS ALL CAPS!!"
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "this is all caps!!")

        test_str = "                   "
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "")

        test_str = "this is all lower space..."
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "this is all lower space...")

        test_str = "  H  e  L    l   O   !"
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "h e l l o !")

        test_str = ""
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "")

        test_str = "........"
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "........")

        test_str = "EX  A M P     LE"
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "ex a m p le")

        test_str = "Test Text Normalization"
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "test text normalization")

        test_str = "AbCd EfGh IjKl MnOp"
        pred = tp.normalize(test_str)
        self.assertEqual(pred, "abcd efgh ijkl mnop")

    def test_no_vowels(self):
        test_str = "This is an example."
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "Ths s n xmpl.")

        test_str = "We love Python!"
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "W lv Pythn!")

        test_str = "AEIOU"
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "")

        test_str = "aeiou"
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "")

        test_str = "QWERTY"
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "QWRTY")

        test_str = "qwerty"
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "qwrty")

        test_str = "AI for ALL!"
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, " fr LL!")

        test_str = "Are there any vowels?"
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "r thr ny vwls?")

        test_str = ""
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "")

        test_str = "abcdefghijklmnopqrstuvwxyz"
        pred = tp.no_vowels(test_str)
        self.assertEqual(pred, "bcdfghjklmnpqrstvwxyz")
