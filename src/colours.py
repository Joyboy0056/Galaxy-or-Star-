class bcolors:
    # Colori di base
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Colori brillanti
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Colori dark/attenuati (usando DIM + colori base)
    DARK_RED = '\033[2;31m'
    DARK_GREEN = '\033[2;32m'
    DARK_YELLOW = '\033[2;33m'
    DARK_BLUE = '\033[2;34m'
    DARK_MAGENTA = '\033[2;35m'
    DARK_CYAN = '\033[2;36m'
    DARK_WHITE = '\033[2;37m'
    DARK_GRAY = '\033[2;90m'
    
    # Colori 256-color per tonalità più scure
    DARK_RED_256 = '\033[38;5;52m'      # Rosso scuro
    DARK_GREEN_256 = '\033[38;5;22m'    # Verde scuro
    DARK_BLUE_256 = '\033[38;5;18m'     # Blu scuro
    DARK_YELLOW_256 = '\033[38;5;94m'   # Giallo scuro/marrone
    DARK_CYAN_256 = '\033[38;5;23m'     # Ciano scuro
    DARK_MAGENTA_256 = '\033[38;5;53m'  # Magenta scuro
    DARK_GRAY_256 = '\033[38;5;236m'    # Grigio scuro
    CHARCOAL = '\033[38;5;240m'         # Carbone
    SLATE = '\033[38;5;244m'            # Ardesia
    
    # Stili
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # Reset
    ENDC = '\033[0m'  # End color
    RESET = '\033[0m'
    
    # Sfondi
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Metodi helper
    @classmethod
    def colored(cls, text, color):
        """Restituisce testo colorato"""
        return f"{color}{text}{cls.ENDC}"
    
    @classmethod
    def success(cls, text):
        """Testo verde per successi"""
        return f"{cls.BRIGHT_GREEN}{text}{cls.ENDC}"
    
    @classmethod
    def error(cls, text):
        """Testo rosso per errori"""
        return f"{cls.BRIGHT_RED}{text}{cls.ENDC}"
    
    @classmethod
    def warning(cls, text):
        """Testo giallo per warning"""
        return f"{cls.BRIGHT_YELLOW}{text}{cls.ENDC}"
    
    @classmethod
    def info(cls, text):
        """Testo blu per info"""
        return f"{cls.BRIGHT_BLUE}{text}{cls.ENDC}"
    
    @classmethod
    def header(cls, text):
        """Testo magenta bold per header"""
        return f"{cls.BOLD}{cls.BRIGHT_MAGENTA}{text}{cls.ENDC}"
    
    @classmethod
    def dark_success(cls, text):
        """Testo verde scuro per successi discreti"""
        return f"{cls.DARK_GREEN_256}{text}{cls.ENDC}"
    
    @classmethod
    def dark_error(cls, text):
        """Testo rosso scuro per errori discreti"""
        return f"{cls.DARK_RED_256}{text}{cls.ENDC}"
    
    @classmethod
    def dark_warning(cls, text):
        """Testo giallo scuro per warning discreti"""
        return f"{cls.DARK_YELLOW_256}{text}{cls.ENDC}"
    
    @classmethod
    def dark_info(cls, text):
        """Testo blu scuro per info discrete"""
        return f"{cls.DARK_BLUE_256}{text}{cls.ENDC}"
    
    @classmethod
    def muted(cls, text):
        """Testo grigio scuro per testi secondari"""
        return f"{cls.DARK_GRAY_256}{text}{cls.ENDC}"

# Esempi d'uso
if __name__ == "__main__":
    # Colori normali vs dark
    print("=== CONFRONTO COLORI ===")
    print(bcolors.success("✅ Successo normale"))
    print(bcolors.dark_success("✅ Successo dark"))
    
    print(bcolors.error("❌ Errore normale"))
    print(bcolors.dark_error("❌ Errore dark"))
    
    print(bcolors.warning("⚠️  Warning normale"))
    print(bcolors.dark_warning("⚠️  Warning dark"))
    
    print(bcolors.info("ℹ️  Info normale"))
    print(bcolors.dark_info("ℹ️  Info dark"))
    
    print(bcolors.header("🎯 HEADER"))
    print(bcolors.muted("Testo secondario/muted"))
    
    print("\n=== PALETTE DARK ===")
    print(f"{bcolors.DARK_RED}Rosso scuro{bcolors.ENDC}")
    print(f"{bcolors.DARK_GREEN}Verde scuro{bcolors.ENDC}")
    print(f"{bcolors.DARK_BLUE}Blu scuro{bcolors.ENDC}")
    print(f"{bcolors.DARK_YELLOW}Giallo scuro{bcolors.ENDC}")
    print(f"{bcolors.DARK_CYAN}Ciano scuro{bcolors.ENDC}")
    print(f"{bcolors.DARK_MAGENTA}Magenta scuro{bcolors.ENDC}")
    print(f"{bcolors.CHARCOAL}Carbone{bcolors.ENDC}")
    print(f"{bcolors.SLATE}Ardesia{bcolors.ENDC}")
    
    print(f"\n{bcolors.BOLD}Testo in grassetto{bcolors.ENDC}")
    print(f"{bcolors.UNDERLINE}Testo sottolineato{bcolors.ENDC}")
    print(bcolors.colored("Testo personalizzato", bcolors.DARK_CYAN_256))