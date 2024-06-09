import re

from spiral import ronin


class CodePreprocessor():
    def __init__(self, stop_words, remove_num=True):
        self.stop_words = stop_words
        self.remove_num = remove_num

    def run(self, text):
        self.text = text
        self.strip_words()
        self.remove_string_literals()
        if self.remove_num:
            self.remove_numbers()
        self.split_camel_case()

        self.to_lower_case()
        self.remove_stop_word()
        return list(filter(None, self.text.split(" ")))

    def strip_words(self):
        for w in ["\n", ";", "[", "]", "}", "{", "(", ")", ",", ".", "\"\""]:
            self.text = self.text.replace(w, " ")


    def remove_string_literals(self):
        self.text = re.sub(r'"[^"]+"', " ", self.text)
        for w in ["\'", "\""]:
            self.text = self.text.replace(w, " ")

    def remove_numbers(self):
        self.text = re.sub(r'0[xX][0-9a-fA-F]+|\d+', " <num> ", self.text)

    def split_camel_case(self):
        self.text = " ".join(ronin.split(self.text))

    def remove_single_word(self):
        self.text = re.sub(r"\b[A-Za-z0-9-_]{1}\b", " ", self.text)

    def to_lower_case(self):
        self.text = self.text.lower()

    def remove_stop_word(self):
        for w in self.stop_words:
            self.text = self.text.replace(w, " ")

