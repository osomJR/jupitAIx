export const commonTranslations = {
  en: {
    back: "Back",
    language: "Language",
    english: "English",
    french: "Français",
    chooseFile: "Choose file",
    fileAccepted: "File accepted",
    formatPolicy: "Format policy",
    previewArea: "Preview area",
    outputFormat: "Output format",
    generating: "Generating...",
    converting: "Converting...",
    translating: "Translating...",
    explaining: "Explaining...",
    summarize: "Summarize",
    explain: "Explain",
    translate: "Translate",
    convert: "Convert",
  },
  fr: {
    back: "Retour",
    language: "Langue",
    english: "English",
    french: "Français",
    chooseFile: "Choisir un fichier",
    fileAccepted: "Fichier accepté",
    formatPolicy: "Règles de format",
    previewArea: "Zone d’aperçu",
    outputFormat: "Format de sortie",
    generating: "Génération...",
    converting: "Conversion...",
    translating: "Traduction...",
    explaining: "Explication...",
    summarize: "Résumer",
    explain: "Expliquer",
    translate: "Traduire",
    convert: "Convertir",
  },
};

export const actionCardTranslations = {
  en: {
    open: "Open",
    accountRequired: "Account required",
  },
  fr: {
    open: "Ouvrir",
    accountRequired: "Compte requis",
  },
};

export const homePageTranslations = {
  en: {
    badge: "Fast AI tools for everyday work",
    heroTitleStart: "Create, refine, and understand content",
    heroTitleHighlight: "in seconds",
    heroDescription:
      "Use powerful AI actions to convert, summarize, correct, translate, and explain content beautifully. Sign in to unlock even more advanced tools.",
    availableNowTitle: "Available now",
    availableNowDescription: "These features are ready for anonymous users.",
    unlockMoreEyebrow: "Unlock more",
    unlockMoreTitle: "Sign in or create an account to access advanced features",
    unlockMoreDescription:
      "Get access to premium AI capabilities like transcription and automatic question generation, designed for deeper workflows and more serious productivity.",
    signIn: "Sign In",
    signUp: "Sign Up",
    advancedFeaturesTitle: "Advanced features",
    advancedFeaturesDescription: "Available after sign in.",
    languageLabel: "Language",
    english: "English",
    french: "Français",
    enabledActions: [
      {
        key: "convert",
        name: "Convert",
        route: "/convert",
        description: "Transform files and content into the format you need.",
      },
      {
        key: "summarize",
        name: "Summarize",
        route: "/summarize",
        description: "Turn long content into sharp, useful highlights.",
      },
      {
        key: "grammar",
        name: "Grammar Correct",
        route: "/grammar",
        description: "Polish your writing with clean, confident corrections.",
      },
      {
        key: "translate",
        name: "Translate",
        route: "/translate",
        description: "Translate text naturally across multiple languages.",
      },
      {
        key: "explain",
        name: "Explain",
        route: "/explain",
        description: "Break down difficult ideas into simple explanations.",
      },
    ],
    lockedActions: [
      {
        key: "transcribe",
        name: "Transcribe",
        route: "/transcribe",
        description: "Convert audio and speech into accurate text.",
      },
      {
        key: "questions",
        name: "Generate Questions",
        route: "/questions",
        description: "Create smart questions from notes, text, or topics.",
      },
      {
        key: "redact",
        name: "Redact",
        route: "/redact",
        description: "XYZ",
      },
      {
        key: "mask",
        name: "Data Mask",
        route: "/mask",
        description: "XYZ",
      },
      {
        key: "compliance",
        name: "Compliance",
        route: "/compliance",
        description: "XYZ",
      },
      {
        key: "extraction",
        name: "Structured Extraction",
        route: "/extraction",
        description: "XYZ",
      },
    ],
  },
  fr: {
    badge: "Des outils IA rapides pour le travail quotidien",
    heroTitleStart: "Créez, améliorez et comprenez le contenu",
    heroTitleHighlight: "en quelques secondes",
    heroDescription:
      "Utilisez de puissantes actions IA pour convertir, résumer, corriger, traduire et expliquer le contenu avec élégance. Connectez-vous pour débloquer encore plus d’outils avancés.",
    availableNowTitle: "Disponible maintenant",
    availableNowDescription:
      "Ces fonctionnalités sont prêtes pour les utilisateurs anonymes.",
    unlockMoreEyebrow: "Débloquez plus",
    unlockMoreTitle:
      "Connectez-vous ou créez un compte pour accéder aux fonctionnalités avancées",
    unlockMoreDescription:
      "Accédez à des capacités IA premium comme la transcription et la génération automatique de questions, conçues pour des flux de travail plus poussés et une productivité plus sérieuse.",
    signIn: "Se connecter",
    signUp: "S’inscrire",
    advancedFeaturesTitle: "Fonctionnalités avancées",
    advancedFeaturesDescription: "Disponibles après connexion.",
    languageLabel: "Langue",
    english: "English",
    french: "Français",
    enabledActions: [
      {
        key: "convert",
        name: "Convertir",
        route: "/convert",
        description:
          "Transformez vos fichiers et contenus dans le format souhaité.",
      },
      {
        key: "summarize",
        name: "Résumer",
        route: "/summarize",
        description:
          "Transformez un long contenu en points clés utiles et précis.",
      },
      {
        key: "grammar",
        name: "Corriger La Grammaire",
        route: "/grammar",
        description:
          "Améliorez votre écriture avec des corrections claires et sûres.",
      },
      {
        key: "translate",
        name: "Traduire",
        route: "/translate",
        description: "Traduisez naturellement du texte dans plusieurs langues.",
      },
      {
        key: "explain",
        name: "Expliquer",
        route: "/explain",
        description: "Décomposez les idées difficiles en explications simples.",
      },
    ],
    lockedActions: [
      {
        key: "transcribe",
        name: "Transcrire",
        route: "/transcribe",
        description:
          "Convertissez l’audio et la parole en texte avec précision.",
      },
      {
        key: "questions",
        name: "Générer Des questions",
        route: "/questions",
        description:
          "Créez des questions intelligentes à partir de notes, de texte ou de sujets.",
      },
      {
        key: "redact",
        name: "Caviarder",
        route: "/redact",
        description: "XYZ",
      },
      {
        key: "mask",
        name: "Masquage Des Données",
        route: "/mask",
        description: "XYZ",
      },
      {
        key: "compliance",
        name: "Conformité",
        route: "/compliance",
        description: "XYZ",
      },
      {
        key: "extraction",
        name: "Extraction Structurée",
        route: "/extraction",
        description: "XYZ",
      },
    ],
  },
};
export const convertPageTranslations = {
  en: {
    badge: "Convert documents and images",
    title: "Convert files across the supported formats you need",
    description:
      "Upload a PDF, Word document, JPG, JPEG, or PNG. Only allowed conversion paths can be selected.",
    uploadTitle: "Upload a file to convert",
    allowedInputs: "Allowed inputs:",
    conversionOutput: "Conversion output",
    previewText:
      "Your conversion result will appear here after you choose a valid input file and an allowed target format.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf, .docx, .jpg, .jpeg, and .png are allowed.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    chooseFileToConvert: "Please choose a file to convert.",
    invalidConversion: "This conversion combination is not allowed.",
    conversionFailed: "Something went wrong while converting the file.",

    conversionCompleted: "Conversion completed.",
    inputFile: "Input file",
    inputExtension: "Input extension",
    outputExtension: "Output extension",
    conversionMatchesRules:
      "The selected conversion matches the allowed document conversion rules.",

    detectedType: "Detected type:",
    from: "From",
    convertTo: "Convert to",
    allowedOutputsFor: "Allowed outputs for",
    none: "none",
    conversionLabel: "Conversion:",

    allowedConversions: "Allowed conversions",
    strictConversionMatrix: "Strict conversion matrix",
    inputCoverage: "Input coverage",
    supportedUploadTypes: "Supported upload types",

    pdfWordTitle: "PDF ↔ Word",
    pdfWordDescription: ".pdf → .docx and .docx → .pdf",
    jpgWordPdfTitle: "JPG / JPEG → PDF or Word",
    jpgWordPdfDescription: ".jpg / .jpeg → .pdf or .docx",
    pngJpgTitle: "PNG → JPG / JPEG",
    pngJpgDescription: ".png → .jpg or .jpeg",

    pdfDocument: "PDF document",
    wordDocument: "Word document",
    jpgImage: "JPG image",
    pngImage: "PNG image",
    unknownFile: "Unknown file",
  },
  fr: {
    badge: "Convertir des documents et des images",
    title: "Convertissez vos fichiers selon les formats pris en charge",
    description:
      "Téléversez un PDF, un document Word, un JPG, un JPEG ou un PNG. Seuls les chemins de conversion autorisés peuvent être sélectionnés.",
    uploadTitle: "Téléversez un fichier à convertir",
    allowedInputs: "Entrées autorisées :",
    conversionOutput: "Résultat de la conversion",
    previewText:
      "Le résultat de la conversion apparaîtra ici après avoir choisi un fichier valide et un format cible autorisé.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les formats .pdf, .docx, .jpg, .jpeg et .png sont autorisés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    chooseFileToConvert: "Veuillez choisir un fichier à convertir.",
    invalidConversion: "Cette combinaison de conversion n’est pas autorisée.",
    conversionFailed:
      "Une erreur s’est produite lors de la conversion du fichier.",

    conversionCompleted: "Conversion terminée.",
    inputFile: "Fichier d’entrée",
    inputExtension: "Extension d’entrée",
    outputExtension: "Extension de sortie",
    conversionMatchesRules:
      "La conversion sélectionnée respecte les règles de conversion autorisées.",

    detectedType: "Type détecté :",
    from: "De",
    convertTo: "Convertir vers",
    allowedOutputsFor: "Sorties autorisées pour",
    none: "aucune",
    conversionLabel: "Conversion :",

    allowedConversions: "Conversions autorisées",
    strictConversionMatrix: "Matrice de conversion stricte",
    inputCoverage: "Couverture des entrées",
    supportedUploadTypes: "Types de fichiers pris en charge",

    pdfWordTitle: "PDF ↔ Word",
    pdfWordDescription: ".pdf → .docx et .docx → .pdf",
    jpgWordPdfTitle: "JPG / JPEG → PDF ou Word",
    jpgWordPdfDescription: ".jpg / .jpeg → .pdf ou .docx",
    pngJpgTitle: "PNG → JPG / JPEG",
    pngJpgDescription: ".png → .jpg ou .jpeg",

    pdfDocument: "Document PDF",
    wordDocument: "Document Word",
    jpgImage: "Image JPG",
    pngImage: "Image PNG",
    unknownFile: "Fichier inconnu",
  },
};
export const explainPageTranslations = {
  en: {
    badge: "Explain content clearly",
    title: "Break down difficult content into simple explanations",
    description:
      "Upload a PDF or Word document, or paste inline text. Unsupported files like PNG are rejected automatically, and the output extension always matches the input extension.",

    fileMode: "Upload file",
    textMode: "Inline text",

    uploadTitle: "Upload content to explain",
    allowedFileInputs:
      "Allowed: .pdf and .docx. Rejected automatically: .png, .jpg, and unsupported formats.",
    outputExtensionWillBe: "Output extension will be",

    pasteTextLabel: "Paste text to explain",
    pasteTextPlaceholder: "Paste or type your text here...",
    inlineTextTreatedAs:
      "Inline text is treated as .txt, so the output extension will also be .txt.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf and .docx uploads are allowed. PNG and other image formats are rejected.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    explanationFailed: "Something went wrong while generating the explanation.",

    generatingExplanation: "Generating explanation...",
    outputFormatLabel: "Output format:",

    policyTitle: "Format policy",
    policySubtitle: "Strict input and output matching",
    allowedUploadsLabel: "Allowed uploads:",
    inlineInputLabel: "Inline input:",
    rejectedAutomaticallyLabel: "Rejected automatically:",
    outputRuleLabel: "Output rule:",
    inlineInputValue: "treated as .txt",
    rejectedAutomaticallyValue: ".png and all unsupported file types",
    outputRuleValue: "output extension must always equal input extension",

    explanationOutputTitle: "Explanation output",
    previewEmpty:
      "Your generated explanation will appear here. The output extension always mirrors the original input extension.",
    outputExtensionLabel: "Output extension:",

    inlineExplanationIntro: "Explanation generated from inline text.",
    fileExplanationIntro: "Explanation generated from {filename}.",
    rewrittenPreviewText:
      "This content has been rewritten into a simpler explanation while keeping the same output format rule.",
    previewLabel: "Preview:",
    inputExtensionLabel: "Input extension:",
    outputExtensionResultLabel: "Output extension:",
    preservedExtensionMessage:
      "The explanation output preserves the same extension as the original uploaded file.",
  },
  fr: {
    badge: "Expliquer clairement le contenu",
    title: "Décomposez les contenus difficiles en explications simples",
    description:
      "Téléversez un PDF ou un document Word, ou collez du texte inline. Les fichiers non pris en charge comme PNG sont rejetés automatiquement, et l’extension de sortie correspond toujours à l’extension d’entrée.",

    fileMode: "Téléverser un fichier",
    textMode: "Texte inline",

    uploadTitle: "Téléverser un contenu à expliquer",
    allowedFileInputs:
      "Autorisés : .pdf et .docx. Rejetés automatiquement : .png, .jpg et les formats non pris en charge.",
    outputExtensionWillBe: "L’extension de sortie sera",

    pasteTextLabel: "Coller le texte à expliquer",
    pasteTextPlaceholder: "Collez ou saisissez votre texte ici...",
    inlineTextTreatedAs:
      "Le texte inline est traité comme .txt, donc l’extension de sortie sera également .txt.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les fichiers .pdf et .docx sont autorisés. Les formats PNG et autres images sont rejetés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    explanationFailed:
      "Une erreur s’est produite lors de la génération de l’explication.",

    generatingExplanation: "Génération de l’explication...",
    outputFormatLabel: "Format de sortie :",

    policyTitle: "Règles de format",
    policySubtitle: "Correspondance stricte entre entrée et sortie",
    allowedUploadsLabel: "Téléversements autorisés :",
    inlineInputLabel: "Entrée inline :",
    rejectedAutomaticallyLabel: "Rejetés automatiquement :",
    outputRuleLabel: "Règle de sortie :",
    inlineInputValue: "traité comme .txt",
    rejectedAutomaticallyValue:
      ".png et tous les types de fichiers non pris en charge",
    outputRuleValue:
      "l’extension de sortie doit toujours être identique à l’extension d’entrée",

    explanationOutputTitle: "Résultat de l’explication",
    previewEmpty:
      "Votre explication générée apparaîtra ici. L’extension de sortie reflète toujours l’extension d’entrée d’origine.",
    outputExtensionLabel: "Extension de sortie :",

    inlineExplanationIntro: "Explication générée à partir du texte inline.",
    fileExplanationIntro: "Explication générée à partir de {filename}.",
    rewrittenPreviewText:
      "Ce contenu a été reformulé en une explication plus simple tout en conservant la même règle de format de sortie.",
    previewLabel: "Aperçu :",
    inputExtensionLabel: "Extension d’entrée :",
    outputExtensionResultLabel: "Extension de sortie :",
    preservedExtensionMessage:
      "Le résultat de l’explication conserve la même extension que le fichier téléversé d’origine.",
  },
};
export const summarizePageTranslations = {
  en: {
    badge: "Summarize content",
    title: "Summarize documents or text with strict format rules",
    description:
      "Upload a PDF or Word document, or paste inline text. Unsupported files like PNG are automatically rejected, and the output extension always matches the input extension.",

    fileMode: "Upload file",
    textMode: "Inline text",

    uploadTitle: "Upload a supported document",
    allowedFileInputs:
      "Allowed: .pdf and .docx. Rejected automatically: .png, .jpg, and all unsupported formats.",
    outputExtensionWillBe: "Output extension will be",

    pasteTextLabel: "Paste text to summarize",
    pasteTextPlaceholder: "Paste or type your text here...",
    inlineTextTreatedAs:
      "Inline text is treated as .txt, so the output extension will also be .txt.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf and .docx uploads are allowed. PNG and other image formats are rejected.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    summaryFailed: "Something went wrong while generating the summary.",

    generatingSummary: "Generating summary...",

    policySubtitle: "Strict input and output matching",
    allowedUploadsLabel: "Allowed uploads:",
    inlineInputLabel: "Inline input:",
    rejectedAutomaticallyLabel: "Rejected automatically:",
    outputRuleLabel: "Output rule:",
    inlineInputValue: "treated as .txt",
    rejectedAutomaticallyValue: ".png and all unsupported file types",
    outputRuleValue: "output extension must always equal input extension",

    summaryOutputTitle: "Summary output",
    previewEmpty:
      "Your generated summary will appear here. The output extension will always mirror the original input extension.",
    outputExtensionLabel: "Output extension:",

    inlineSummaryIntro: "Summary generated from inline text.",
    fileSummaryIntro: "Summary generated from {filename}.",
    translatedPreviewLabel: "Preview:",
    inputExtensionLabel: "Input extension:",
    outputExtensionResultLabel: "Output extension:",
    preservedExtensionMessage:
      "The output format remains the same as the uploaded file format.",
  },
  fr: {
    badge: "Résumer le contenu",
    title:
      "Résumez des documents ou du texte avec des règles de format strictes",
    description:
      "Téléversez un PDF ou un document Word, ou collez du texte inline. Les fichiers non pris en charge comme PNG sont automatiquement rejetés, et l’extension de sortie correspond toujours à l’extension d’entrée.",

    fileMode: "Téléverser un fichier",
    textMode: "Texte inline",

    uploadTitle: "Téléverser un document pris en charge",
    allowedFileInputs:
      "Autorisés : .pdf et .docx. Rejetés automatiquement : .png, .jpg et tous les formats non pris en charge.",
    outputExtensionWillBe: "L’extension de sortie sera",

    pasteTextLabel: "Coller le texte à résumer",
    pasteTextPlaceholder: "Collez ou saisissez votre texte ici...",
    inlineTextTreatedAs:
      "Le texte inline est traité comme .txt, donc l’extension de sortie sera également .txt.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les fichiers .pdf et .docx sont autorisés. Les formats PNG et autres images sont rejetés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    summaryFailed: "Une erreur s’est produite lors de la génération du résumé.",

    generatingSummary: "Génération du résumé...",

    policySubtitle: "Correspondance stricte entre entrée et sortie",
    allowedUploadsLabel: "Téléversements autorisés :",
    inlineInputLabel: "Entrée inline :",
    rejectedAutomaticallyLabel: "Rejetés automatiquement :",
    outputRuleLabel: "Règle de sortie :",
    inlineInputValue: "traité comme .txt",
    rejectedAutomaticallyValue:
      ".png et tous les types de fichiers non pris en charge",
    outputRuleValue:
      "l’extension de sortie doit toujours être identique à l’extension d’entrée",

    summaryOutputTitle: "Résultat du résumé",
    previewEmpty:
      "Votre résumé généré apparaîtra ici. L’extension de sortie reflétera toujours l’extension d’entrée d’origine.",
    outputExtensionLabel: "Extension de sortie :",

    inlineSummaryIntro: "Résumé généré à partir du texte inline.",
    fileSummaryIntro: "Résumé généré à partir de {filename}.",
    translatedPreviewLabel: "Aperçu :",
    inputExtensionLabel: "Extension d’entrée :",
    outputExtensionResultLabel: "Extension de sortie :",
    preservedExtensionMessage:
      "Le format de sortie reste identique à celui du fichier téléversé.",
  },
};

export const translatePageTranslations = {
  en: {
    badge: "Translate content naturally",
    title: "Translate documents or text while preserving format rules",
    description:
      "Upload a PDF or Word document, or paste inline text. Unsupported files like PNG are rejected automatically, and the output extension always matches the input extension.",

    fileMode: "Upload file",
    textMode: "Inline text",

    targetLanguageLabel: "Translate to",
    targetLanguagePlaceholder:
      "Enter any target language, e.g. French, Yoruba, Japanese",
    targetLanguageHelp:
      "Enter any target language supported by the model instead of choosing from a limited dropdown.",

    uploadTitle: "Upload content to translate",
    allowedFileInputs:
      "Allowed: .pdf and .docx. Rejected automatically: .png, .jpg, and unsupported formats.",
    outputExtensionWillBe: "Output extension will be",

    pasteTextLabel: "Paste text to translate",
    pasteTextPlaceholder: "Paste or type your text here...",
    inlineTextTreatedAs:
      "Inline text is treated as .txt, so the output extension will also be .txt.",

    unsupportedFileType:
      "Unsupported file type: {ext}. Only .pdf and .docx uploads are allowed. PNG and other image formats are rejected.",
    fileTooLarge: "File is too large. Maximum allowed size is {maxSize} MB.",
    targetLanguageRequired: "Please enter a target language.",
    translationFailed: "Something went wrong while generating the translation.",

    generatingTranslation: "Generating translation...",

    policySubtitle: "Strict input and output matching",
    allowedUploadsLabel: "Allowed uploads:",
    inlineInputLabel: "Inline input:",
    rejectedAutomaticallyLabel: "Rejected automatically:",
    outputRuleLabel: "Output rule:",
    inlineInputValue: "treated as .txt",
    rejectedAutomaticallyValue: ".png and all unsupported file types",
    outputRuleValue: "output extension must always equal input extension",

    translationOutputTitle: "Translation output",
    previewEmpty:
      "Your generated translation will appear here. The output extension always mirrors the original input extension.",
    outputExtensionLabel: "Output extension:",

    inlineTranslationIntro: "Translation generated from inline text.",
    fileTranslationIntro: "Translation generated from {filename}.",
    targetLanguageResultLabel: "Target language:",
    translatedPreviewLabel: "Translated preview:",
    inputExtensionLabel: "Input extension:",
    outputExtensionResultLabel: "Output extension:",
    preservedExtensionMessage:
      "The translated output keeps the same extension as the uploaded input.",

    languageSuggestions: [
      "English",
      "French",
      "Spanish",
      "German",
      "Portuguese",
      "Brazilian Portuguese",
      "Arabic",
      "Chinese",
      "Simplified Chinese",
      "Traditional Chinese",
      "Japanese",
      "Korean",
      "Hindi",
      "Yoruba",
      "Hausa",
      "Igbo",
      "Swahili",
      "Turkish",
      "Russian",
      "Italian",
      "Dutch",
    ],
  },
  fr: {
    badge: "Traduire le contenu naturellement",
    title:
      "Traduisez des documents ou du texte tout en préservant les règles de format",
    description:
      "Téléversez un PDF ou un document Word, ou collez du texte inline. Les fichiers non pris en charge comme PNG sont automatiquement rejetés, et l’extension de sortie correspond toujours à l’extension d’entrée.",

    fileMode: "Téléverser un fichier",
    textMode: "Texte inline",

    targetLanguageLabel: "Traduire vers",
    targetLanguagePlaceholder:
      "Saisissez une langue cible, par ex. français, yoruba, japonais",
    targetLanguageHelp:
      "Saisissez toute langue cible prise en charge par le modèle au lieu de choisir dans une liste limitée.",

    uploadTitle: "Téléverser un contenu à traduire",
    allowedFileInputs:
      "Autorisés : .pdf et .docx. Rejetés automatiquement : .png, .jpg et les formats non pris en charge.",
    outputExtensionWillBe: "L’extension de sortie sera",

    pasteTextLabel: "Coller le texte à traduire",
    pasteTextPlaceholder: "Collez ou saisissez votre texte ici...",
    inlineTextTreatedAs:
      "Le texte inline est traité comme .txt, donc l’extension de sortie sera également .txt.",

    unsupportedFileType:
      "Type de fichier non pris en charge : {ext}. Seuls les fichiers .pdf et .docx sont autorisés. Les formats PNG et autres images sont rejetés.",
    fileTooLarge:
      "Le fichier est trop volumineux. La taille maximale autorisée est de {maxSize} Mo.",
    targetLanguageRequired: "Veuillez saisir une langue cible.",
    translationFailed:
      "Une erreur s’est produite lors de la génération de la traduction.",

    generatingTranslation: "Génération de la traduction...",

    policySubtitle: "Correspondance stricte entre entrée et sortie",
    allowedUploadsLabel: "Téléversements autorisés :",
    inlineInputLabel: "Entrée inline :",
    rejectedAutomaticallyLabel: "Rejetés automatiquement :",
    outputRuleLabel: "Règle de sortie :",
    inlineInputValue: "traité comme .txt",
    rejectedAutomaticallyValue:
      ".png et tous les types de fichiers non pris en charge",
    outputRuleValue:
      "l’extension de sortie doit toujours être identique à l’extension d’entrée",

    translationOutputTitle: "Résultat de la traduction",
    previewEmpty:
      "Votre traduction générée apparaîtra ici. L’extension de sortie reflète toujours l’extension d’entrée d’origine.",
    outputExtensionLabel: "Extension de sortie :",

    inlineTranslationIntro: "Traduction générée à partir du texte inline.",
    fileTranslationIntro: "Traduction générée à partir de {filename}.",
    targetLanguageResultLabel: "Langue cible :",
    translatedPreviewLabel: "Aperçu traduit :",
    inputExtensionLabel: "Extension d’entrée :",
    outputExtensionResultLabel: "Extension de sortie :",
    preservedExtensionMessage:
      "Le résultat traduit conserve la même extension que l’entrée téléversée.",

    languageSuggestions: [
      "English",
      "French",
      "Spanish",
      "German",
      "Portuguese",
      "Brazilian Portuguese",
      "Arabic",
      "Chinese",
      "Simplified Chinese",
      "Traditional Chinese",
      "Japanese",
      "Korean",
      "Hindi",
      "Yoruba",
      "Hausa",
      "Igbo",
      "Swahili",
      "Turkish",
      "Russian",
      "Italian",
      "Dutch",
    ],
  },
};
export const transcribePageTranslations = {
  en: {
    title: "Transcribe Audio/Video",
    submit: "Transcribe",
    processing: "Processing...",
    output: "Transcription Output",
    noFile: "Please upload a file.",
    invalidFileType: "Invalid file type. Allowed: .mp3, .mp4, .mkv, .mov",
    error: "Something went wrong during transcription.",
    noOutput: "No transcription result returned.",
  },
  fr: {
    title: "Transcrire Audio/Vidéo",
    submit: "Transcrire",
    processing: "Traitement...",
    output: "Résultat de transcription",
    noFile: "Veuillez télécharger un fichier.",
    invalidFileType:
      "Type de fichier invalide. Autorisés : .mp3, .mp4, .mkv, .mov",
    error: "Une erreur est survenue pendant la transcription.",
    noOutput: "Aucun résultat de transcription retourné.",
  },
};
export const grammarCorrectPageTranslations = {
  en: {
    title: "Grammar Correct",
    submit: "Correct Grammar",
    processing: "Processing...",
    output: "Corrected Output",
    noInput: "Please upload a file or enter text.",
    invalidFileType:
      "Invalid file type. Allowed: .pdf, .docx, or inline text only.",
    error: "Something went wrong during grammar correction.",
    noOutput: "No result returned.",
    fileProcessed:
      "Your file was processed successfully. The output format matches the input format.",
    inlineTextLabel: "Inline Text (.txt)",
    inlineTextPlaceholder: "Paste or type your text here...",
    uploadLabel: "Upload File",
    uploadHelp:
      "Allowed file types: .pdf, .docx. Inline text is treated as .txt.",
  },
  fr: {
    title: "Correction Grammaticale",
    submit: "Corriger la grammaire",
    processing: "Traitement...",
    output: "Résultat corrigé",
    noInput: "Veuillez télécharger un fichier ou saisir du texte.",
    invalidFileType:
      "Type de fichier invalide. Autorisés : .pdf, .docx ou texte en ligne uniquement.",
    error: "Une erreur est survenue pendant la correction grammaticale.",
    noOutput: "Aucun résultat retourné.",
    fileProcessed:
      "Votre fichier a été traité avec succès. Le format de sortie correspond au format d’entrée.",
    inlineTextLabel: "Texte en ligne (.txt)",
    inlineTextPlaceholder: "Collez ou saisissez votre texte ici...",
    uploadLabel: "Téléverser un fichier",
    uploadHelp:
      "Types de fichiers autorisés : .pdf, .docx. Le texte en ligne est traité comme .txt.",
  },
};