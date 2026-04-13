


import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    // 1. On récupère le texte de la réclamation (claimText) envoyé par le front
    const { claimText } = await request.json();

    if (!claimText) {
      return NextResponse.json({ error: 'Le texte de la réclamation est requis.' }, { status: 400 });
    }

    // 2. APPEL API PYTHON DANS DOCKER
    const aiResponse = await fetch('http://127.0.0.1:8000/tags', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_claim: claimText }),
    });

    if (!aiResponse.ok) {
      throw new Error(`L'API Python Docker ne répond pas (Status: ${aiResponse.status})`);
    }

    const aiData = await aiResponse.json();

    // 3. ON RENVOIE LE RÉSULTAT
    return NextResponse.json({ category: aiData.prediction });

  } catch (error) {
    console.error('Erreur lors de la classification locale :', error);
    return NextResponse.json(
      { error: 'Erreur lors de la communication avec l\'API Docker ZenAssist.' },
      { status: 500 }
    );
  }
}










