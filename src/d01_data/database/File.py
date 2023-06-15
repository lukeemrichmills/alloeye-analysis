

class File:
    def __init__(self, filename, pID, practice, block, data_type, appendix):
        self.filename = filename
        self.pID = pID
        self.practice = practice
        self.block = block
        self.data_type = data_type
        self.appendix = appendix

    def __str__(self):
        return f'File({self.filename})'
