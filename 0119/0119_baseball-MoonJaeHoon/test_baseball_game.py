# -*- coding: utf8 -*-

import unittest
import baseball_game as bg

from mock import patch
from io import StringIO


class TestBaseballGame(unittest.TestCase):

    def test_is_digit(self):
        self.assertEqual(True, bg.is_digit("3232"))
        self.assertEqual(False, bg.is_digit("32.2"))
        self.assertEqual(False, bg.is_digit("15.4"))
        self.assertEqual(True, bg.is_digit("323"))

    def test_is_between_100_and_999(self):
        self.assertEqual(True, bg.is_between_100_and_999("100"))
        self.assertEqual(False, bg.is_between_100_and_999("5"))
        self.assertEqual(False, bg.is_between_100_and_999("4934"))
        self.assertEqual(True, bg.is_between_100_and_999("503"))
        for i in range(1, 99):
            self.assertEqual(False, bg.is_between_100_and_999(str(i)))
        for i in range(100, 999):
            self.assertEqual(True, bg.is_between_100_and_999(str(i)))
        for i in range(1000, 100000):
            self.assertEqual(False, bg.is_between_100_and_999(str(i)))

    def test_is_duplicated_number(self):
        self.assertEqual(True, bg.is_duplicated_number("100"))
        self.assertEqual(True, bg.is_duplicated_number("110"))
        self.assertEqual(True, bg.is_duplicated_number("111"))
        self.assertEqual(True, bg.is_duplicated_number("220"))
        self.assertEqual(False, bg.is_duplicated_number("312"))
        self.assertEqual(False, bg.is_duplicated_number("542"))
        for i in range(100, 999):
            id_duplicated = bg.is_duplicated_number(str(i))
            self.assertEqual(id_duplicated, bg.is_duplicated_number(str(i)))

    def test_is_validated_number(self):
        self.assertEqual(False, bg.is_validated_number("100"))
        self.assertEqual(False, bg.is_validated_number("99999"))
        self.assertEqual(False, bg.is_validated_number("efkqk"))
        self.assertEqual(False, bg.is_validated_number("19"))
        self.assertEqual(True, bg.is_validated_number("567"))
        self.assertEqual(True, bg.is_validated_number("154"))
        self.assertEqual(True, bg.is_validated_number("437"))
        self.assertEqual(False, bg.is_validated_number("110"))
        self.assertEqual(False, bg.is_validated_number("111"))


    def test_get_not_duplicated_three_digit_number(self):
        for i in range(5000):
            is_duplicated = self.is_duplicated_number(
                str(bg.get_not_duplicated_three_digit_number()))
            self.assertEqual(False, is_duplicated)

    def test_get_strikes_or_ball(self):
        strikes, balls = bg.get_strikes_or_ball("123", "123")
        self.assertEqual(3, strikes)
        self.assertEqual(0, balls)

        strikes, balls = bg.get_strikes_or_ball("456", "123")
        self.assertEqual(0, strikes)
        self.assertEqual(0, balls)

        strikes, balls = bg.get_strikes_or_ball("312", "123")
        self.assertEqual(0, strikes)
        self.assertEqual(3, balls)

        strikes, balls = bg.get_strikes_or_ball("472", "764")
        self.assertEqual(0, strikes)
        self.assertEqual(2, balls)

        strikes, balls = bg.get_strikes_or_ball("174", "175")
        self.assertEqual(2, strikes)
        self.assertEqual(0, balls)

    def test_is_yes(self):
        self.assertEqual(True, bg.is_yes("yEs"))
        self.assertEqual(True, bg.is_yes("yES"))
        self.assertEqual(True, bg.is_yes("Y"))
        self.assertEqual(True, bg.is_yes("y"))
        self.assertEqual(True, bg.is_yes("yes"))
        self.assertEqual(True, bg.is_yes("YES"))

        self.assertEqual(False, bg.is_yes("n01"))
        self.assertEqual(False, bg.is_yes("n3493"))
        self.assertEqual(False, bg.is_yes("no"))
        self.assertEqual(False, bg.is_yes("yyyyyyy"))
        self.assertEqual(False, bg.is_yes("yesyesyes"))

    def test_is_no(self):
        self.assertEqual(True, bg.is_no("no"))
        self.assertEqual(True, bg.is_no("NO"))
        self.assertEqual(True, bg.is_no("No"))
        self.assertEqual(True, bg.is_no("nO"))
        self.assertEqual(True, bg.is_no("n"))
        self.assertEqual(True, bg.is_no("N"))

        self.assertEqual(False, bg.is_no("n01"))
        self.assertEqual(False, bg.is_no("non"))
        self.assertEqual(False, bg.is_no("nnnnnnn"))
        self.assertEqual(False, bg.is_no("nonono"))

        self.assertEqual(False, bg.is_no("YES"))

    def test_main(self):
        for x in range(2000):
            with patch('builtins.input', side_effect=["0"]):
                with patch('sys.stdout', new=StringIO()) as fakeOutput:
                    bg.main()
                    console = fakeOutput.getvalue().strip().split("\n")
                    random_number = console[1][-3:].strip()
                    self.assertFalse(
                        self.is_duplicated_number(random_number))

        with patch('builtins.input', side_effect=["woe", "ewe", "121", "545", "0"]):
            with patch('sys.stdout', new=StringIO()) as fakeOutput:
                bg.main()
                console = fakeOutput.getvalue().strip().split("\n")
                for i in range(2,6):
                    self.assertTrue("WRONG INPUT" in console[i].upper())

        input_list = [str(value) for value in range(101, 1000)]
        input_list.append("YES")
        input_list.extend([str(value) for value in range(101, 1000)])
        input_list.append("no0")
        input_list.append("n23")
        input_list.append("nvjd")
        input_list.append("nown")
        input_list.append("3orio2kr3o")

        input_list.append("No")

        with patch('builtins.input', side_effect=input_list):
            with patch('sys.stdout', new=StringIO()) as fakeOutput:
                bg.main()
                random_number = []
                console = fakeOutput.getvalue().strip().split("\n")
                for line in console:
                    if "RANDOM NUMBER" in line.upper():
                        random_number.append(line[-3:].strip())
                target_number = random_number[0]
                for i in range(0, 899):
                    if int(input_list[i]) < int(target_number):
                        if self.is_duplicated_number(input_list[i]):
                            self.assertTrue(
                                "WRONG INPUT" in console[i + 2].upper())
                        else:
                            strikes, ball = self.get_strikes_or_ball(
                                input_list[i], target_number)
                            self.assertIn(
                                str(strikes), console[i + 2].upper())
                            self.assertIn(
                                str(ball), console[i + 2].upper())
                    elif int(input_list[i]) > int(target_number):
                        self.assertTrue(
                                "WRONG INPUT" in console[i + 2].upper())
                    elif int(input_list[i]) == int(target_number):
                        self.assertIn(str(3), console[i + 2].upper())
                        self.assertIn(str(0), console[i + 2].upper())
                        self.assertIn(
                            "Strikes".upper(), console[i + 2].upper())
                        self.assertIn(
                            "Balls".upper(), console[i + 2].upper())

                target_number = random_number[1]
                for i in range(900, len(input_list)):
                    if input_list[i].isdigit():
                        if int(input_list[i]) < int(target_number):
                            if self.is_duplicated_number(input_list[i]):
                                self.assertTrue(
                                    "WRONG INPUT" in console[i + 2].upper())
                            else:
                                strikes, ball = self.get_strikes_or_ball(
                                    input_list[i], target_number)
                                self.assertIn(
                                    str(strikes), console[i + 2].upper())
                                self.assertIn(
                                    str(ball), console[i + 2].upper())
                        elif int(input_list[i]) > int(target_number):
                            self.assertTrue(
                                    "WRONG INPUT" in console[i + 2].upper())
                        elif int(input_list[i]) == int(target_number):
                            self.assertIn(str(3), console[i + 2].upper())
                            self.assertIn(str(0), console[i + 2].upper())
                            self.assertIn(
                                "Strikes".upper(), console[i + 2].upper())
                            self.assertIn(
                                "Balls".upper(), console[i + 2].upper())
                    else:
                        if not(self.is_no(input_list[i])):
                            self.assertTrue(
                                "WRONG INPUT" in console[i + 2].upper())
                        else:
                            self.assertIn(
                                "Thank you".upper(), console[i + 2].upper())
                            self.assertIn(
                                "End of the Game".upper(), console[i + 3].upper())


    def is_no(self, one_more_input):
        if one_more_input.upper() == 'NO':
            return True
        if one_more_input.upper() == 'N':
            return True
        return False


    def is_duplicated_number(self, three_digit):
        for number in three_digit:
            if three_digit.count(number) > 1:
                return True
        return False

    def get_strikes_or_ball(self, user_input_number, random_number):
        result = []
        if random_number == user_input_number:
            result = [3, 0]

        strikes = 0
        ball = 0

        for number in user_input_number:
            if (number in random_number):
                if user_input_number.index(number) is random_number.index(number):
                    strikes += 1
                else:
                    ball += 1
        result = [strikes, ball]
        return result
