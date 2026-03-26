import { NextResponse } from 'next/server';
import { Mistral } from '@mistralai/mistralai';

// initialisation Mistral
const apiKey = process.env.MISTRAL_API_KEY;

export async function POST(request: Request) {
  try {
    const { claimText } = await request.json();

    if (!claimText) {
      return NextResponse.json({ error: 'Le texte de la réclamation est requis.' }, { status: 400 });
    }

    if (!apiKey) {
      return NextResponse.json({ error: 'La clé API Mistral n\'est pas configurée.' }, { status: 500 });
    }

    const client = new Mistral({ apiKey });


    const categories_str = "1. Debt collection\n2. Consumer Loan\n3. Credit card or prepaid card\n4. Mortgage\n5. Vehicle loan or lease\n6. Money transfer, virtual currency, or money service\n7. Money transfers\n8. Other (TO BE UPDATED)";

    const prompt = `Tu es un expert en tri de réclamations client chez ZenAssist. 
    Classe la réclamation ci-dessous dans l'UNE de ces catégories : ${categories_str}
    
    CONSIGNE STRICTE : Réponds uniquement par le nom de la catégorie, rien d'autre.
    
    Réclamation : ${claimText}`;

    const chatResponse = await client.chat.complete({
      model: 'mistral-large-latest',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
    });

    // Extraction de la réponse 
    const content = chatResponse.choices?.[0]?.message?.content;
    const expectedCategory = typeof content === 'string' ? content.trim() : '';

    return NextResponse.json({ category: expectedCategory });
  } catch (error) {
    console.error('Erreur API Mistral :', error);
    return NextResponse.json({ error: 'Erreur lors de la communication avec Mistral AI.' }, { status: 500 });
  }
}
