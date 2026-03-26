import { useState } from "react";
import { Sparkles } from "lucide-react";
import styles from "./AutoTag.module.css";

interface AutoTagProps {
    claimContent: string;
    onTagGenerated: (tag: string) => void;
}

export default function AutoTag({ claimContent, onTagGenerated }: AutoTagProps) {
    const [isUpdating, setIsUpdating] = useState(false);

    const handleAutoTag = async () => {
        setIsUpdating(true);
        try {
            const response = await fetch('/api/categorize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ claimText: claimContent }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Erreur inconnue lors du taggage automatique');
            }

            const data = await response.json();
            if (data.category) {
                onTagGenerated(data.category);
            }
        } catch (error) {
            console.error('Erreur API Mistral :', error);
            alert('Impossible de générer le tag automatiquement.');
        } finally {
            setIsUpdating(false);
        }
    };

    return (
        <div className={styles.container}>
            <button
                onClick={handleAutoTag}
                disabled={isUpdating}
                className={styles.button}
            >
                {isUpdating ? (
                    <span className={styles.updating}>Auto-tagging...</span>
                ) : (
                    <>
                        <Sparkles size={18} />
                        Auto-tag IA
                    </>
                )}
            </button>
        </div>
    );
}