import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_merge import classify, read_rows

class AuditTests(unittest.TestCase):
    def test_multigenre_is_retained(self):
        self.assertEqual(classify({'Genres':'Pop/Rock, Electronic', 'Styles':'House'})[:2], (True, True))
    def test_substring_is_not_exact_token(self):
        self.assertFalse(classify({'Genres':'Electronic Rock', 'Styles':'House'})[0])
    def test_style_alone_and_broad_dance_tags(self):
        self.assertFalse(classify({'Genres':'Pop/Rock', 'Styles':'Electronic, House'})[1])
        self.assertEqual(classify({'Genres':'Electronic', 'Styles':'Club/Dance, Dance-Pop'})[:2], (True, False))
    def test_duplicate_ids_fail(self):
        raw = b'Song,Artist,Title,Genres,Styles,Quadrant\nA,a,t,Electronic,House,Q1\nA,b,u,Electronic,House,Q2\n'
        with self.assertRaises(ValueError):
            read_rows(raw)

if __name__ == '__main__':
    unittest.main()
