LANGUAGE_TAG = {
"c" : "C",
"c++" : " C++",
"cpp" : " C++",
"c" : " C",
"csharp" : " C",
"c-sharp" : " C",
"php" : " PHP",
"js" : " JavaScript",
"javascript" : " JavaScript",
"typescript" : " TypeScript",
"go" : " Go",
"shell" : " Shell",
"rust" : " Rust",
"sql" : "SQL",
"kotlin" : " Kotlin",
"vb" : "' Visual Basic",
"ruby" : " Ruby",
"pascal" : " Pascal",
"r" : " R",
"cuda" : " Cuda",
"dart" : " Dart",
"lua" : " Lua",
"objectivec" : " Objective-C",
"objective-c" : " Objective-C",
"objective-c++": " Objective-C++",
"python" : " Python",
"perl" : " Perl",
"prolog" : f"% Prolog",
"swift" : " swift",
"lisp" : "; Lisp",
"java" : " Java",
"scala" : " Scala",
"tex" : f"% TeX",
"vue" : "Vue",
"markdown" : "Markdown",
"html" : "HTML",
"fortran" : "!Fortran",
"lean" : "Lean",
"matlab" : f"% Matlab",
"delphi" : "{Delphi}",
"scheme" : "; Scheme",
"basic" : "' Basic",
"assembly" : "; Assembly",
"groovy" : " Groovy",
"gdscript" : " GDScript",
"haskell" : "Haskell",
"julia" : " Julia",
"elixir" : " Elixir",
"excel" : "' Excel",
"clojure" : "; Clojure",
"actionscript" : " ActionScript",
"solidity" : " Solidity",
"powershell" : " PowerShell",
"cobol" : " Cobol",
"awk" : " AWK",
"sparql" : " SPARQL",
"augeas" : " Augeas",
"cmake" : " CMake",
"f-sharp" : " F",
"stan" : " Stan",
"isabelle" : "(*Isabelle*)",
"dockerfile" : " Dockerfile",
"rmarkdown" : " RMarkdown",
"literate-agda": "Literate Agda",
"tcl" : " Augeas",
"glsl" : " GLSL",
"antlr" : " ANTLR",
"verilog" : " Verilog",
"racket" : "; Racket",
"standard-ml" : "(*language:Standard ML*)",
"elm" : "Elm",
"yaml" : " YAML",
"smalltalk" : "'' Smalltalk",
"idris" : "Idris",
"visual-basic" : "' Visual Basic",
"protocol-buffer": " Protocol Buffer",
"bluespec" : " Bluespec",
"applescript" : "AppleScript",
"makefile" : " Makefile",
"tcsh" : " TCSH",
"maple" : " Maple",
"systemverilog": " SystemVerilog",
"literate-coffeescript": " Literate CoffeeScript",
"vhdl" : "VHDL",
"java-server-pages": " Java Server Pages",
"coffeescript" : " CoffeeScript",
"emacs-lisp" : "; Emacs Lisp",
"mathematica" : " Mathematica",
"xslt" : "XSLT",
"zig" : " Zig",
"common-lisp" : "; Common Lisp"
}


class NaiveFilter:
    def __init__(self):
        self.sensitivewds = set([])

    def parse(self, path):
        for keyword in open(path):
            self.sensitivewds.add(keyword.strip().encode('utf-8').decode('utf-8').lower())

    def filter(self, message, replace="[[*]]"):
        message = str(message).lower()
        for kw in self.keywords:
            message = message.replace(kw, replace)
        return message


# you could add the word you want to replace
word_dict = {
    " ":" ",
}
