import os
import ast


class CornellData:
    def __init__(self, dir):
        '''
        Args:
            dir (string): directory where to load the corpus
        '''
        self.lines = {}
        self.conversations = []

        MOVIE_LINES_FIELDS = ["lineID", "characterID",
                              "movieID", "character", "text"]
        MOVIE_CONVERSATIONS_FIELDS = [
            "character1ID", "character2ID", "movieID", "utteranceIDs"]

        self.lines = self.loadLines(os.path.join(
            dir, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        self.conversations = self.loadConversations(os.path.join(
            dir, "movie_conversations.txt"), MOVIE_CONVERSATIONS_FIELDS)

    def loadLines(self, file, fields):
        '''
        Args:
            file (str): file to load
            field (set<str>): fields to extract
        Return:
            dict<dict<str>>: the extracted fields for each line
        '''
        lines = {}

        with open(file, 'r', encoding='iso-8859-1', errors='ignore') as f:
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]

                lines[lineObj['lineID']] = lineObj

        return lines

    def loadConversations(self, file, fields):
        """
        Args:
            file (str): file to load
            field (set<str>): fields to extract
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        conversations = []

        with open(file, 'r', encoding='iso-8859-1', errors='ignore') as f:
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                # Convert string to list
                # (convObj["utteranceIDs"] =="['L598485', 'L598486', ...]")
                lineIds = ast.literal_eval(convObj["utteranceIDs"])

                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId])

                conversations.append(convObj)

        return conversations

    def getConversations(self):
        return self.conversations
