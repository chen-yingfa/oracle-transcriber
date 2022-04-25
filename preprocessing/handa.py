import json
from typing import Union, List

import regex as re

import utils


ZH_PUNCS = '※■□*。，…… …:;<>《》!！?？；：“”‘’\{\}〔〕()（）【】［］｛｝～·．－—'
DATA_PATH = '/data/private/songchenyang/hanzi_filter/handa'
PATH_META = DATA_PATH + '/H/oracle_meta_H.json'


def remove_html(s: str) -> str:
    return re.sub(r'<.*?>', '', s)


def is_punc(s: str) -> bool:
    return s in ZH_PUNCS


def remove_punc(s: str):
    for p in ZH_PUNCS:
        s = s.replace(p, '')
    return s


def has_dup(s: str) -> bool:
    chars = set(s)
    return len(chars) < len(s)


def count_dup():
    meta = json.load(open(PATH_META, 'r'))
    count = 0
    total = len(meta)
    target_book_name = 'H00185'
    for record in meta:
        # print(record['book_name'])
        if record['book_name'] == target_book_name:
            print(record['modern_text'])
            print(hanzis)
            exit()
            continue
        hanzis = remove_html(record['modern_text'])
        hanzis = remove_punc(hanzis)
        if has_dup(hanzis):
            count += 1
            # print(record['book_name'])
            # print(record['row_order'])
            # print(hanzis, set(hanzis), len(hanzis), len(set(hanzis)))
            
    print(count)
    print(total)


def get_all_book_names() -> set:
    meta = json.load(open(PATH_META, 'r'))
    names = set()
    for record in meta:
        book_name = record['book_name']
        names.add(book_name)
    return names


class Handa:
    '''
    Currently ignores all books with duplicate Chinese characters.
    '''
    def __init__(self):
        self.meta = json.load(open(PATH_META, 'r'))
        self.book_dict = self.gen_dict(self.meta)
        self.not_found = []
        
    def gen_dict(self, meta):
        count = 0
        book_dict = {}
        for record in meta:
            book_name = record['book_name']
            row_order = record['row_order']
            hanzis = ''
            char_files = []
            for ch in record['r_chars']:
                if ch['char'] in ZH_PUNCS:
                    continue
                if len(ch['char']) != 1:
                    continue
                hanzis += ch['char']
                char_files.append(ch['img'])
                
            assert len(hanzis) == len(char_files)
            count += len(char_files)
            if book_name not in book_dict:
                book_dict[book_name] = {}
            book_dict[book_name][str(row_order)] = {
                'hanzis': hanzis,
                'hanzis_full': record['modern_text'],
                'char_files': char_files,
            }
        print('Total number of valid characters:', count)

        # Record books that have no characters or duplicate characters
        total_num_rows = sum([len(book) for book in book_dict.values()])
        total_num_books = len(book_dict)
        self.books_with_dup = []
        self.books_empty = []
        for book_name, book in book_dict.items():
            hanzis = ''
            for row_order, row in book.items():
                hanzis += row['hanzis']
            if len(hanzis) == 0:
                self.books_empty.append(book_name)
            elif has_dup(hanzis):
                self.books_with_dup.append(book_name)
        
        # for book_name in self.books_with_dup:
        #     del book_dict[book_name]
        for book_name in self.books_empty:
            del book_dict[book_name]

        # Save files
        utils.dump_json(book_dict, 'handa-stats/books.json')
        utils.dump_json(self.books_with_dup, 'handa-stats/books_with_dup.json')
        utils.dump_json(self.books_empty, 'handa-stats/books_empty.json')
                
        print(f'Total number of books: {total_num_books}')
        print(f'Total number of rows: {total_num_rows}')
        print(f'Books with duplicate characters: {len(self.books_with_dup)}')
        print(f'Books with no characters: {len(self.books_empty)}')
        print(f'Final number of books: {len(book_dict)}')
        
        count = 0
        for book in book_dict.values():
            for row in book.values():
                count += len(row['char_files'])
        print('Total number of characters:', count)
        return book_dict
        
    def get_all_files(self, book_name: str) -> List[str]:
        '''Return all character image files of a book'''
        if book_name not in self.book_dict:
            return None
        all_files = []
        for row_meta in self.book_dict[book_name].values():
            all_files += row_meta['char_files']
            print(row_meta['hanzis'])
        for i in range(len(all_files)):
            all_files[i] = DATA_PATH + '/H/characters/' + all_files[i]
        return all_files
        
    def get_file(self, book_name: str, hanzi: str) -> Union[None, str]:
        '''
        Get the file path of the image of the oracle bone script characters
        of the given hanzi in the given book.
        '''
        if book_name not in self.book_dict:
            return None
        
        # Check if this hanzi occurs multiple times in the book, if so, skip
        all_hanzis = ''
        for row_meta in self.book_dict[book_name].values():
            all_hanzis += row_meta['hanzis']
        
        occ = 0
        for c in all_hanzis:
            if c == hanzi:
                occ += 1
                if occ > 1: break
        
        # If not in the book or multiple occurence
        if occ != 1:
            hanzis = ''.join([r['hanzis_full'] for r in self.book_dict[book_name].values()])
            self.not_found.append([hanzi, book_name, hanzis])
            return None
        
        # Now we know the hanzi occurs exactly once
        for row_meta in self.book_dict[book_name].values():
            hanzis = row_meta['hanzis']
            if hanzi in hanzis:
                idx = hanzis.index(hanzi)
                file = row_meta['char_files'][idx]                    
                return DATA_PATH + '/H/characters/' + file
            
        raise ValueError('Should not reach here')

    # def is_book_have_dup(self, book_name: str) -> bool:
    #     return book_name in self.books_with_dup


if __name__ == '__main__':
    count_dup()
