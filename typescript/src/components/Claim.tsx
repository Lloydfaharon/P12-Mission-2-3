'use client';

import styles from './Claim.module.css';
import { Claim } from '@/database/queries.ts';
import AutoTag from './AutoTag.tsx';
import { updateClaimTag } from '@/api-client.ts';

interface ClaimProps {
  claim: Claim;
  isSelected: boolean;
  onClick: () => void;
  onTagClick?: (tag: string) => void;
  onTagUpdate?: (claimId: number, tag: string) => void;
}

export default function ClaimComponent({ claim, isSelected, onClick, onTagClick, onTagUpdate }: ClaimProps) {
  const handleTagClick = (e: React.MouseEvent) => {
    e.stopPropagation();

    if (claim.tag && onTagClick) {
      onTagClick(claim.tag);
    }
  };

  const handleAutoTagGenerated = async (tag: string) => {
    try {
      await updateClaimTag(claim.id, tag);
      if (onTagUpdate) {
        onTagUpdate(claim.id, tag);
      }
    } catch (error) {
      console.error('Erreur lors de la sauvegarde du tag :', error);
    }
  };

  return (
    <div
      className={`${styles.container} ${isSelected ? styles.selected : ''}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      aria-pressed={isSelected}
    >
      <div className={styles.content}>
        <p
          className={styles.text}
          id={`claim-content-${claim.id}`}
        >
          {claim.content}
        </p>
        
        {/* On gère les événements de clics sur AutoTag de manière indépendante pour ne pas déclencher onClick de la carte entière */}
        <div style={{ marginTop: '12px' }} onClick={(e) => e.stopPropagation()}>
          {claim.tag ? (
            <button
              className={styles.tag}
              onClick={handleTagClick}
              aria-label={`Navigate to ${claim.tag} inbox`}
              title={`Go to ${claim.tag} inbox`}
            >
              {claim.tag}
            </button>
          ) : (
            <AutoTag 
              claimContent={claim.content} 
              onTagGenerated={handleAutoTagGenerated} 
            />
          )}
        </div>
      </div>
    </div>
  );
}