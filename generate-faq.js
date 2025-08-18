const fs = require('fs');
const path = require('path');

class FAQGenerator {
    constructor() {
        // You can expand these keyword maps based on your content
        this.keywordMaps = {
            en: {
                'ethics': ['suffering', 'pain', 'cruel', 'moral', 'ethics', 'kill', 'slaughter', 'respect', 'honor'],
                'health': ['healthy', 'disease', 'protein', 'iron', 'b12', 'vitamin', 'nutrition', 'supplement'],
                'environment': ['environment', 'climate', 'emission', 'water', 'land', 'rainforest', 'soy', 'amazon'],
                'social': ['culture', 'tradition', 'religion', 'family', 'friends', 'social', 'outsider', 'choice'],
                'myths': ['brain', 'canine', 'teeth', 'omnivore', 'ancestor', 'natural', 'lion', 'apex'],
                'practical': ['hard', 'expensive', 'taste', 'boring', 'supplement', 'food', 'cooking', 'restaurant'],
                'animals': ['cow', 'pig', 'chicken', 'sheep', 'animal', 'farm', 'milk', 'egg', 'wool', 'honey']
            },
            de: {
                'ethik': ['leiden', 'schmerz', 'grausam', 'moral', 'ethik', 'töten', 'schlachten', 'respekt', 'ehre'],
                'gesundheit': ['gesund', 'krankheit', 'protein', 'eisen', 'b12', 'vitamin', 'ernährung', 'supplement'],
                'umwelt': ['umwelt', 'klima', 'emission', 'wasser', 'land', 'regenwald', 'soja', 'amazon'],
                'sozial': ['kultur', 'tradition', 'religion', 'familie', 'freunde', 'sozial', 'außenseiter', 'wahl'],
                'mythen': ['gehirn', 'eckzahn', 'zähne', 'allesfresser', 'vorfahren', 'natürlich', 'löwe', 'apex'],
                'praktisch': ['schwer', 'teuer', 'geschmack', 'langweilig', 'supplement', 'essen', 'kochen', 'restaurant'],
                'tiere': ['kuh', 'schwein', 'huhn', 'schaf', 'tier', 'bauernhof', 'milch', 'ei', 'wolle', 'honig']
            },
            nl: {
                'ethiek': ['lijden', 'pijn', 'wreed', 'moraal', 'ethiek', 'doden', 'slachten', 'respect', 'eer'],
                'gezondheid': ['gezond', 'ziekte', 'proteïne', 'ijzer', 'b12', 'vitamine', 'voeding', 'supplement'],
                'milieu': ['milieu', 'klimaat', 'uitstoot', 'water', 'land', 'regenwoud', 'soja', 'amazone'],
                'sociaal': ['cultuur', 'traditie', 'religie', 'familie', 'vrienden', 'sociaal', 'buitenstaander', 'keuze'],
                'mythen': ['brein', 'hoektand', 'tanden', 'omnivoor', 'voorouders', 'natuurlijk', 'leeuw', 'apex'],
                'praktisch': ['moeilijk', 'duur', 'smaak', 'saai', 'supplement', 'eten', 'koken', 'restaurant'],
                'dieren': ['koe', 'varken', 'kip', 'schaap', 'dier', 'boerderij', 'melk', 'ei', 'wol', 'honing']
            }
        };

    }

    // Generate tags based on text content
    generateTags(text, language = 'en') {
        const tags = new Set();
        const lowerText = text.toLowerCase();
        const keywordMap = this.keywordMaps[language] || this.keywordMaps.en;

        // Add content-based category tags
        Object.entries(keywordMap).forEach(([category, keywords]) => {
            if (keywords.some(keyword => lowerText.includes(keyword))) {
                tags.add(category);
            }
        });

        return Array.from(tags);
    }

    // Process single language data
    processLanguageData(data, language) {
        return data.map((item, index) => {
            const combinedText = `${item.argument || ''} ${item.antwort || ''}`;
            const contentTags = this.generateTags(combinedText, language);

            // Create comprehensive tags array
            const allTags = [...contentTags];

            // Add fallback tag if no content tags were found
            if (contentTags.length === 0) {
                allTags.push(language === 'de' ? 'allgemein' : language === 'nl' ? 'algemeen' : 'general');
            }

            return {
                id: index + 1,
                question: item.argument || '',
                answer: item.antwort || '',
                tags: allTags
            };
        });
    }

    // Detect if an argument is actually a counter-argument based on patterns
    detectCounterArgument(text, language = 'en') {
        const counterPatterns = {
            en: ['vegans', 'vegan', 'plants have feelings', 'personal choice', 'too expensive', 'too hard'],
            de: ['veganer', 'vegan', 'pflanzen haben gefühle', 'persönliche wahl', 'zu teuer', 'zu schwer'],
            nl: ['veganisten', 'veganist', 'planten hebben gevoelens', 'persoonlijke keuze', 'te duur', 'te moeilijk']
        };

        const patterns = counterPatterns[language] || counterPatterns.en;
        const lowerText = text.toLowerCase();

        return patterns.some(pattern => lowerText.includes(pattern));
    }

    // Main generation method
    async generateFAQFiles(inputConfig) {
        const {
            originalFile,
            outputDir = './faq-output',
            languages = ['en', 'de', 'nl'],
            translatedFiles = {} // e.g., { de: './arguments_de.json', nl: './arguments_nl.json' }
        } = inputConfig;

        // Create output directory
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        const results = {};

        for (const lang of languages) {
            try {
                let sourceFile;

                if (lang === 'en' || !translatedFiles[lang]) {
                    sourceFile = originalFile;
                } else {
                    sourceFile = translatedFiles[lang];
                }

                console.log(`Processing ${lang} from: ${sourceFile}`);

                // Read source data
                const rawData = fs.readFileSync(sourceFile, 'utf8');
                const data = JSON.parse(rawData);

                // Process and enhance data
                const processedData = this.processLanguageData(data, lang);

                // Write to output file
                const outputFile = path.join(outputDir, `faq_${lang}.json`);
                fs.writeFileSync(outputFile, JSON.stringify(processedData, null, 2));

                results[lang] = {
                    file: outputFile,
                    count: processedData.length
                };

                console.log(`✅ Generated ${outputFile} with ${processedData.length} items`);

            } catch (error) {
                console.error(`Error processing ${lang}:`, error.message);
                results[lang] = { error: error.message };
            }
        }

        // Generate summary file
        const summaryFile = path.join(outputDir, 'faq_summary.json');
        const summary = {
            generated: new Date().toISOString(),
            languages: results,
            totalItems: Object.values(results)
                .filter(r => !r.error)
                .reduce((sum, r) => sum + (r.count || 0), 0)
        };

        fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));
        console.log('\n📊 Generation Summary:');
        console.log(JSON.stringify(summary, null, 2));

        return results;
    }
}

// Usage example and configuration
async function main() {
    const generator = new FAQGenerator();

    // Configure your file paths
    const config = {
        originalFile: './antworten.json', // Your vegan arguments file
        outputDir: './faq-data',
        languages: ['en', 'de', 'nl'],
        translatedFiles: {
            de: './antworten_de.json', // Your German translation
            nl: './antworten_nl.json'  // Your Dutch translation
            // If these don't exist, it will use the original file
        }
    };

    try {
        await generator.generateFAQFiles(config);
        console.log('\n🎉 FAQ files generated successfully!');

        console.log('\nNext steps:');
        console.log('1. Check the generated files in ./faq-data/');
        console.log('2. Review the auto-generated tags');
        console.log('3. Manually refine tags if needed');
        console.log('4. Update your FAQ component to load these files');

    } catch (error) {
        console.error('Generation failed:', error);
    }
}

// Utility function to add this to your package.json scripts
function getPackageScriptEntry() {
    return {
        "generate-faq": "node generate-faq.js"
    };
}

// Export for use in build systems
module.exports = { FAQGenerator, main };

// Run if called directly
if (require.main === module) {
    main();
}