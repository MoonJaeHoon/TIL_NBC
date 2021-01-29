# -*- coding: utf8 -*-

import unittest
import random
import text_processing2 as tp


class TestTextProcessing(unittest.TestCase):
    def test_digits_to_words(self):
        test_str = ""
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "")

        test_str = "Zip Code: 19104"
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "one nine one zero four")

        test_str = "Pi is 3.1415..."
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "three one four one five")

        test_str = "There are 3 jellies."
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "three")

        test_str = "Your room number is 205."
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "two zero five")

        test_str = "1588-1588"
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "one five eight eight one five eight eight")

        test_str = "My number: 010-1234-5678"
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "zero one zero one two three four five six seven eight")

        test_str = "No digits"
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "")

        test_str = "No digits here too!?$%"
        pred = tp.digits_to_words(test_str)
        self.assertEqual(pred, "")

    def test_to_camel_case(self):
        test_str = "to_camel_case"
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "toCamelCase")

        test_str = "__EXAMPLE__NAME__"
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "exampleName")

        test_str = "alreadyCamel"
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "alreadyCamel")

        test_str = "_______"
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "")

        test_str = ""
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "")

        test_str = "not_Camel_Yet"
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "notCamelYet")

        test_str = "NOT_CAMEL_Yet"
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "notCamelYet")

        test_str = "abc_def_ghi"
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "abcDefGhi")

        test_str = "     "
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, "     ")

        test_str = "....."
        pred = tp.to_camel_case(test_str)
        self.assertEqual(pred, ".....")
